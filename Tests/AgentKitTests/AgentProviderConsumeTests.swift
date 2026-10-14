import Foundation
import Synchronization
import Testing

@testable import AgentKit

/// Serves one scripted streaming response per test for host `bounded.test`.
private final class BoundedURLProtocol: URLProtocol, @unchecked Sendable {
    struct State: Sendable {
        var chunks: [Data] = []
        var contentType = "text/event-stream"
        var leaveOpen = false
        var stopped: AsyncStream<Void>.Continuation?
    }

    static let state = Mutex(State())

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "bounded.test" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let fixture = Self.state.withLock { $0 }
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": fixture.contentType]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        for chunk in fixture.chunks { client?.urlProtocol(self, didLoad: chunk) }
        if !fixture.leaveOpen { client?.urlProtocolDidFinishLoading(self) }
    }

    override func stopLoading() {
        let stopped = Self.state.withLock { $0.stopped }
        stopped?.yield(())
        stopped?.finish()
    }
}

private func client(
    _ body: String, contentType: String = "text/event-stream", leaveOpen: Bool = false,
    stopped: AsyncStream<Void>.Continuation? = nil
) -> AgentProviderClient {
    BoundedURLProtocol.state.withLock {
        $0 = .init(chunks: [Data(body.utf8)], contentType: contentType, leaveOpen: leaveOpen, stopped: stopped)
    }
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [BoundedURLProtocol.self]
    configuration.timeoutIntervalForRequest = 5
    configuration.timeoutIntervalForResource = 5
    return AgentProviderClient(
        provider: ModelProvider(
            name: "Bounded", apiFormat: .chatCompletions, inferenceURL: "https://bounded.test/v1/chat"),
        model: AIModel(id: "m"), secret: "s", session: URLSession(configuration: configuration)
    )
}

private func chunk(_ text: String) -> String {
    #"data: {"choices":[{"index":0,"delta":{"content":"\#(text)"}}]}"# + "\n\n"
}

private let finish = #"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"# + "\n\ndata: [DONE]\n\n"
private let request = AgentModelRequest(systemPrompt: "Summarize", messages: [], tools: [])

@Suite(.serialized)
struct AgentProviderConsumeTests {
    @Test func deliversEventsInOrderOnTheCallingTaskAndRequiresATerminal() async throws {
        var text = ""
        var finished: AgentStopReason?
        try await client(chunk("A") + chunk("B") + finish).consume(request) { event in
            switch event {
            case .textDelta(let delta): text += delta
            case .finished(let reason): finished = reason
            default: break
            }
        }
        #expect(text == "AB" && finished == .completed)
        await #expect(throws: AgentProviderError.self) {
            try await client(chunk("A")).consume(request) { _ in }
        }
    }

    @Test func boundsTotalBytesLinesAndEvents() async throws {
        let many = (0..<20).map { _ in chunk("x") }.joined() + finish
        await #expect(throws: AgentProviderError.self) {
            try await client(many).consume(request, limits: .init(maximumTransportBytes: 200)) { _ in }
        }
        let longLine = chunk(String(repeating: "y", count: 300_000)) + finish
        await #expect(throws: SSEStreamError.self) {
            try await client(longLine).consume(request) { _ in }
        }
        var delivered = 0
        await #expect(throws: AgentProviderError.self) {
            try await client(many).consume(request, limits: .init(maximumEvents: 3)) { _ in delivered += 1 }
        }
        #expect(delivered == 3)
        // A whole JSON body is capped the same way.
        let json =
            #"{"choices":[{"index":0,"message":{"role":"assistant","content":"\#(String(repeating: "z", count: 500))"},"finish_reason":"stop"}]}"#
        await #expect(throws: AgentProviderError.self) {
            try await client(json, contentType: "application/json").consume(
                request, limits: .init(maximumTransportBytes: 100)
            ) { _ in }
        }
    }

    @Test func aThrowingCallbackStopsReadingAndCancelsTheTransfer() async throws {
        struct Refused: Error {}
        let stop = AsyncStream<Void>.makeStream()
        let open = client(chunk("A") + chunk("B"), leaveOpen: true, stopped: stop.continuation)
        var seen = 0
        await #expect(throws: Refused.self) {
            try await open.consume(request) { _ in
                seen += 1
                throw Refused()
            }
        }
        #expect(seen == 1)
        var iterator = stop.stream.makeAsyncIterator()
        _ = await iterator.next()  // the open transfer was cancelled
    }

    @Test func streamAndCompleteKeepTheirBehavior() async throws {
        let body = chunk("A") + finish
        let events = try await client(body).complete(request)
        #expect(events.contains { if case .textDelta("A") = $0 { true } else { false } })
        var streamed = ""
        for try await event in client(body).stream(request) {
            if case .textDelta(let delta) = event { streamed += delta }
        }
        #expect(streamed == "A")
    }
}
