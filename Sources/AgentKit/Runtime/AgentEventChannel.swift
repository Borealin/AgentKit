import Foundation
import Synchronization

/// The one ordered path every agent event takes, whoever raised it.
///
/// Two things used to publish events independently: `AgentRuntime`, from inside
/// its actor, and `AgentApprovalBroker`, through a closure the caller handed it.
/// Each wrapped its durable write in its own unstructured `Task`, and separate
/// tasks have no order between them — so a `toolFinished` could be written
/// before the `toolStarted` it follows. Sequential tool calls hid most of it;
/// parallel ones would not.
///
/// `yield` on an `AsyncStream.Continuation` is ordered and safe from any
/// isolation, which is exactly what a fan-in point needs when the writers are a
/// task group of concurrently running tools. Durable events are handed to a
/// single drain task that awaits each write before starting the next, so
/// storage sees them in the order they happened rather than in whatever order
/// the scheduler got around to them.
public nonisolated final class AgentEventChannel: Sendable {
    /// The UI's sequence. Single-consumer, like every `AsyncStream`: one channel
    /// belongs to one run.
    public let events: AsyncStream<AgentEvent>

    private let output: AsyncStream<AgentEvent>.Continuation
    private enum Write: Sendable {
        case event(AgentEvent, CommitReceipt?)
        case fence(CommitReceipt)
    }
    private let durable: AsyncStream<Write>.Continuation
    private let recorder: Task<Void, Never>
    private let admission = Mutex(false)
    private let hostCommits: Bool

    /// `committing` lets a host await its own authoritative write. The UI
    /// sequence remains independently consumable, so it can supply that ACK.
    public init(
        repository: any AgentRunPersisting,
        committing: (@Sendable (AgentEvent) async throws -> Void)? = nil
    ) {
        let (events, output) = AsyncStream<AgentEvent>.makeStream()
        let (durableEvents, durable) = AsyncStream<Write>.makeStream()
        self.events = events
        self.output = output
        self.durable = durable
        hostCommits = committing != nil
        recorder = Task {
            var failure: (any Error)?
            for await write in durableEvents {
                switch write {
                case .event(let event, let receipt):
                    if failure == nil {
                        do {
                            if event.isDurable, let runID = event.associatedRunID {
                                try await repository.record(event, runID: runID)
                            }
                            try await committing?(event)
                        } catch { failure = error }
                    }
                    if let receipt {
                        if let failure {
                            receipt.resolve(.failure(failure))
                        } else {
                            receipt.resolve(.success(())) { output.yield(event) }
                        }
                    }
                case .fence(let receipt):
                    receipt.resolve(failure.map { .failure($0) } ?? .success(()))
                }
            }
        }
    }

    deinit {
        output.finish()
        durable.finish()
        // The drain ends on its own once its stream finishes; cancelling here
        // would abandon writes that were already accepted.
    }

    public func emit(_ event: AgentEvent) {
        admission.withLock { closed in
            guard !closed else { return }
            output.yield(event)
            // Streaming text is deliberately not durable: the transcript is written
            // from run snapshots, so a reply must not also stream database writes.
            // See `AgentEvent.isDurable`.
            if event.isDurable || (hostCommits && event.isHostCommitBoundary) {
                durable.yield(.event(event, nil))
            }
        }
    }

    /// Waits for every durable event admitted before this call. The first
    /// write failure stays latched for this run; later fences fail too.
    public func flush() async throws {
        let receipt = CommitReceipt()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                receipt.install(continuation)
                admission.withLock { closed in
                    guard !closed else {
                        receipt.resolve(.failure(CancellationError()))
                        return
                    }
                    durable.yield(.fence(receipt))
                }
            }
        } onCancel: {
            receipt.resolve(.failure(CancellationError()))
        }
    }

    /// Publishes a terminal event only after its write succeeds. In contrast
    /// to `emit`, the host must not wait for this event's UI delivery to ACK it.
    public func emitCommitted(_ event: AgentEvent) async throws {
        let receipt = CommitReceipt()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                receipt.install(continuation)
                admission.withLock { closed in
                    guard !closed else {
                        receipt.resolve(.failure(CancellationError()))
                        return
                    }
                    durable.yield(.event(event, receipt))
                }
            }
        } onCancel: {
            receipt.resolve(.failure(CancellationError()))
        }
    }

    /// Ends the UI sequence and lets the recorder finish what it was given.
    public func finish() {
        admission.withLock { closed in
            closed = true
            output.finish()
            durable.finish()
        }
    }
}

extension AgentEvent {
    fileprivate var isHostCommitBoundary: Bool {
        switch self {
        case .messageFinished, .steeringDelivered: true
        default: false
        }
    }
}

/// Cancelling a fence stops waiting, not an already accepted atomic write.
/// A late completion cannot resume its waiter twice or publish cancelled success.
private final class CommitReceipt: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, any Error>?
    private var result: Result<Void, any Error>?

    func install(_ continuation: CheckedContinuation<Void, any Error>) {
        let completed: Result<Void, any Error>? = lock.withLock {
            if let result = self.result { return result }
            self.continuation = continuation
            return nil
        }
        if let completed { continuation.resume(with: completed) }
    }

    func resolve(_ result: Result<Void, any Error>, beforeResume: () -> Void = {}) {
        let waiter = lock.withLock {
            guard self.result == nil else { return nil as CheckedContinuation<Void, any Error>? }
            self.result = result
            beforeResume()
            let waiter = continuation
            continuation = nil
            return waiter
        }
        waiter?.resume(with: result)
    }
}
