import Foundation
import Testing

@testable import AgentKit

private enum CommitTestFailure: Error { case write }

private actor CommitProbe {
    var requests = 0
    var executed: [String] = []
    var toolNames: Set<String> = []
    func request(_ tools: [AgentToolDescriptor]) -> Int {
        toolNames = Set(tools.map(\.name))
        defer { requests += 1 }
        return requests
    }
    func execute(_ name: String) { executed.append(name) }
}

private struct CommitModel: AgentModelStreaming {
    let probe: CommitProbe
    func stream(_ request: AgentModelRequest) -> AsyncThrowingStream<AgentModelStreamEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                if await probe.request(request.tools) == 0 {
                    for name in ["first", "second"] {
                        continuation.yield(
                            .toolCallSnapshot(id: name, providerItemID: nil, name: name, arguments: "{}"))
                    }
                    continuation.yield(.finished(.toolCalls))
                } else {
                    continuation.yield(.textDelta("done"))
                    continuation.yield(.finished(.completed))
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

private func commitTools(_ probe: CommitProbe) -> [AnyAgentTool] {
    ["first", "second"].map { name in
        AnyAgentTool(
            descriptor: AgentToolDescriptor(
                name: name, summary: name,
                inputSchema: .object(["type": .string("object")]), target: .local, approvalPolicy: .approve),
            execute: { invocation, _ in
                await probe.execute(name)
                return AgentToolResult(callID: invocation.call.id, content: name)
            })
    }
}

private func commitRun(
    repository: any AgentRunPersisting = InMemoryAgentRunRepository(), probe: CommitProbe,
    includesSharedScratchTools: Bool = true,
    committing: (@Sendable (AgentEvent) async throws -> Void)? = nil
) async -> [AgentEvent] {
    let channel = AgentEventChannel(repository: repository, committing: committing)
    var configuration = AgentLoopConfiguration()
    configuration.toolExecution = .sequential
    configuration.includesSharedScratchTools = includesSharedScratchTools
    let runtime = AgentRuntime(
        model: CommitModel(probe: probe), tools: commitTools(probe),
        approval: AgentApprovalBroker(reviewer: nil, manualApproval: { _ in .allow }),
        repository: repository, configuration: configuration, channel: channel)
    var events: [AgentEvent] = []
    for await event in await runtime.start(
        AgentRunRequest(conversationID: UUID(), prompt: "go", permissionMode: .fullAccess))
    {
        events.append(event)
    }
    return events
}

private func committedOutcome(_ events: [AgentEvent]) -> AgentRunState? {
    events.compactMap { if case .runFinished(let run) = $0 { run.state } else { nil } }.last
}

@Suite struct AgentCommitTests {
    @Test(arguments: [true, false])
    func hostCanKeepAnExplicitCatalogWithoutChangingTheDefault(shared: Bool) async {
        let probe = CommitProbe()
        let events = await commitRun(probe: probe, includesSharedScratchTools: shared)
        #expect(committedOutcome(events) == .completed)
        #expect(await probe.toolNames.contains("scratch_read") == shared)
        #expect(await probe.toolNames.isSuperset(of: ["first", "second"]))
    }

    @Test func failedMessageCommitStopsBothTools() async {
        let probe = CommitProbe()
        let events = await commitRun(probe: probe) { event in
            if case .messageFinished = event { throw CommitTestFailure.write }
        }
        #expect(await probe.executed.isEmpty)
        #expect(await probe.requests == 1)
        #expect(committedOutcome(events) == .failed)
    }

    @Test func failedToolResultCommitStopsTheNextToolAndModelRequest() async {
        let probe = CommitProbe()
        let events = await commitRun(probe: probe) { event in
            if case .toolFinished = event { throw CommitTestFailure.write }
        }
        #expect(await probe.executed == ["first"])
        #expect(await probe.requests == 1)
        #expect(committedOutcome(events) == .failed)
    }

    @Test func failedDispatchIntentNeverRunsTheTool() async {
        let probe = CommitProbe()
        let events = await commitRun(probe: probe) { event in
            if case .toolStarted = event { throw CommitTestFailure.write }
        }
        #expect(await probe.executed.isEmpty)
        #expect(committedOutcome(events) == .failed)
    }

    @Test func terminalCommitFailureCannotPublishSuccess() async {
        let probe = CommitProbe()
        let events = await commitRun(probe: probe) { event in
            if case .runFinished(let run) = event, run.state == .completed { throw CommitTestFailure.write }
        }
        #expect(await probe.executed == ["first", "second"])
        #expect(committedOutcome(events) == .failed)
        #expect(!events.contains { if case .runFinished(let run) = $0 { run.state == .completed } else { false } })
    }

    @Test func snapshotFailurePreventsTheFirstModelRequest() async {
        let probe = CommitProbe()
        let events = await commitRun(repository: FailingSnapshotRepository(), probe: probe)
        #expect(await probe.requests == 0)
        #expect(await probe.executed.isEmpty)
        #expect(committedOutcome(events) == .failed)
    }

    @Test func successfulFencesPreserveResultAndTerminalOrder() async {
        let probe = CommitProbe()
        let events = await commitRun(probe: probe)
        #expect(await probe.executed == ["first", "second"])
        #expect(await probe.requests == 2)
        #expect(committedOutcome(events) == .completed)
        let results = events.compactMap { if case .toolFinished(_, let result) = $0 { result.callID } else { nil } }
        #expect(results == ["first", "second"])
    }

    @Test func cancelledFenceDoesNotCancelOrPoisonTheAcceptedWrite() async throws {
        let gate = CommitTestGate()
        let entered = CommitTestGate()
        let repository = InMemoryAgentRunRepository()
        let channel = AgentEventChannel(repository: repository) { _ in
            await entered.open()
            await gate.wait()
        }
        let invocation = AgentToolInvocation(
            runID: UUID(), call: AgentToolCall(id: "call", name: "read", arguments: .object([:])))
        channel.emit(.toolStarted(invocation))
        await entered.wait()
        let waiter = Task { try await channel.flush() }
        waiter.cancel()
        // Always release the fake write, even if cancellation regresses.
        let release = Task {
            try? await Task.sleep(for: .milliseconds(100))
            await gate.open()
        }
        switch await waiter.result {
        case .success: Issue.record("A cancelled fence reported success")
        case .failure(let error): #expect(error is CancellationError)
        }
        await release.value
        try await channel.flush()
        channel.finish()
    }

    @Test func runtimeCancellationSavesAndReportsCancelled() async throws {
        let repository = InMemoryAgentRunRepository()
        let runtime = AgentRuntime(
            model: CancellableCommitModel(), tools: [],
            approval: AgentApprovalBroker(reviewer: nil, manualApproval: { _ in .allow }),
            repository: repository)
        var events: [AgentEvent] = []
        let conversationID = UUID()
        for await event in await runtime.start(
            AgentRunRequest(conversationID: conversationID, prompt: "go", permissionMode: .fullAccess))
        {
            events.append(event)
            if case .messageDelta = event { await runtime.cancel() }
        }
        #expect(committedOutcome(events) == .cancelled)
        let stored = await repository.loadConversations().first { $0.id == conversationID }
        #expect(stored?.lastRunState == .cancelled)
    }

    @Test func cancellationWhileMessageCommitWaitsSettlesEveryCallWithoutDispatch() async throws {
        let entered = CommitTestGate()
        let release = CommitTestGate()
        let probe = CommitProbe()
        let repository = InMemoryAgentRunRepository()
        let channel = AgentEventChannel(repository: repository) { event in
            if case .messageFinished = event {
                await entered.open()
                await release.wait()
            }
        }
        let runtime = AgentRuntime(
            model: CommitModel(probe: probe), tools: commitTools(probe),
            approval: AgentApprovalBroker(reviewer: nil, manualApproval: { _ in .allow }),
            repository: repository, channel: channel)
        let conversationID = UUID()
        let events = await runtime.start(
            AgentRunRequest(conversationID: conversationID, prompt: "go", permissionMode: .fullAccess))
        let reader = Task { () -> [AgentEvent] in
            var collected: [AgentEvent] = []
            for await event in events { collected.append(event) }
            return collected
        }
        await entered.wait()
        await runtime.cancel()
        await release.open()
        let collected = await reader.value
        let results = collected.compactMap {
            if case .toolFinished(_, let result) = $0 { result } else { nil as AgentToolResult? }
        }
        #expect(await probe.executed.isEmpty)
        #expect(await probe.requests == 1)
        #expect(committedOutcome(collected) == .cancelled)
        #expect(results.map(\.callID) == ["first", "second"])
        #expect(results.allSatisfy { $0.isError && $0.content == AgentInterruptedToolCall.notStartedReason })
        let stored = try #require(await repository.loadConversations().first { $0.id == conversationID })
        #expect(stored.lastRunState == .cancelled)
        #expect(stored.messages.map(\.role) == [.user, .assistant, .tool, .tool])
        #expect(stored.messages.suffix(2).map(\.toolCallID) == ["first", "second"])
    }
}

private struct CancellableCommitModel: AgentModelStreaming {
    func stream(_ request: AgentModelRequest) -> AsyncThrowingStream<AgentModelStreamEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                continuation.yield(.textDelta("partial"))
                do {
                    try await Task.sleep(for: .seconds(30))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

private actor CommitTestGate {
    private var opened = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        if opened { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func open() {
        opened = true
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }
}

private struct FailingSnapshotRepository: AgentRunPersisting {
    let base = InMemoryAgentRunRepository()
    func save(_ snapshot: AgentRunSnapshot) async throws { throw CommitTestFailure.write }
    func record(_ event: AgentEvent, runID: UUID) async throws { await base.record(event, runID: runID) }
    func persistCompaction(_ record: AgentCompactionRecord, conversationID: UUID) async throws {
        try await base.persistCompaction(record, conversationID: conversationID)
    }
    func loadConversations() async throws -> [StoredAgentConversation] { await base.loadConversations() }
    func deleteConversation(_ id: UUID) async throws { await base.deleteConversation(id) }
    func setTitle(_ title: String, conversationID: UUID) async throws {
        await base.setTitle(title, conversationID: conversationID)
    }
    func setConversationMetadata(_ metadata: StoredAgentConversationMetadata, conversationID: UUID) async throws {
        try await base.setConversationMetadata(metadata, conversationID: conversationID)
    }
    func reorderConversations(_ updates: [StoredAgentConversationOrder]) async throws {
        try await base.reorderConversations(updates)
    }
    func truncateConversation(
        _ id: UUID, fromUserMessageID messageID: UUID, replacingWith message: AgentTranscriptMessage
    ) async throws {
        try await base.truncateConversation(id, fromUserMessageID: messageID, replacingWith: message)
    }
    func createConversation(_ conversation: StoredAgentConversation) async throws {
        try await base.createConversation(conversation)
    }
}
