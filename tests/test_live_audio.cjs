const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {
    StreamingResampler, PCMPacketizer, encodePCM16, sendPCM, MicrophoneStream,
} = require('../app/static/js/live_audio.js');

function convert(input, from, to, chunks) {
    const converter = new StreamingResampler(from, to);
    const output = [];
    let offset = 0, chunk = 0;
    while (offset < input.length) {
        const size = chunks[chunk++ % chunks.length];
        output.push(...converter.process(input.subarray(offset, offset + size)));
        offset += size;
    }
    output.push(...converter.flush());
    return output;
}

for (const [from, to] of [[44100, 16000], [44100, 24000], [48000, 16000], [48000, 24000], [16000, 16000]]) {
    test(`resampler preserves duration and waveform over 12 seconds: ${from} -> ${to}`, () => {
        const input = Float32Array.from({length: from * 12 + 37}, (_, i) => Math.sin(2 * Math.PI * 173 * i / from) * 0.6);
        const output = convert(input, from, to, [128, 127, 1024, 73, 2048]);
        const expectedLength = Math.ceil(input.length * to / from);
        assert.equal(output.length, expectedLength);
        let maxError = 0;
        // Compare independently to the continuous input time coordinate, including every block boundary.
        for (let i = 0; i < output.length; i++) {
            const pos = i * from / to;
            const left = Math.min(Math.floor(pos), input.length - 1);
            const right = Math.min(left + 1, input.length - 1);
            const expected = input[left] + (input[right] - input[left]) * (pos - left);
            maxError = Math.max(maxError, Math.abs(output[i] - expected));
        }
        assert.ok(maxError < 1e-6, `phase drift/error: ${maxError}`);
    });
}

test('resampling is independent of callback segmentation, including 1-sample callbacks', () => {
    const input = Float32Array.from({length: 11029}, (_, i) => Math.sin(i / 23));
    const whole = convert(input, 44100, 16000, [input.length]);
    const chunked = convert(input, 44100, 16000, [1, 2, 7, 128]);
    assert.equal(whole.length, chunked.length);
    assert.ok(whole.every((value, i) => Math.abs(value - chunked[i]) < 1e-6));
});

test('PCM is signed little-endian with clipping, silence and finite-value protection', () => {
    const bytes = encodePCM16(Float32Array.of(-2, -1, -0.5, 0, 0.5, 1, 2, NaN));
    assert.deepEqual(Array.from(new Int16Array(bytes)), [-32768, -32768, -16384, 0, 16384, 32767, 32767, 0]);
    assert.deepEqual(Array.from(new Uint8Array(bytes).slice(0, 2)), [0, 128]);
});

test('packetizer sends 20ms packets and flushes the final short word tail exactly once', () => {
    const sent = [];
    const packetizer = new PCMPacketizer(320, buffer => sent.push(buffer));
    const audio = Float32Array.from({length: 977}, (_, i) => (i - 490) / 1000);
    for (let i = 0; i < audio.length; i += 128) packetizer.push(audio.subarray(i, i + 128));
    packetizer.flush();
    packetizer.flush();
    assert.deepEqual(sent.map(packet => packet.byteLength), [640, 640, 640, 34]);
    assert.deepEqual(Buffer.concat(sent.map(packet => Buffer.from(packet))), Buffer.from(encodePCM16(audio)));
});

test('backpressure fails visibly before dropping or queueing more speech', () => {
    const sent = [];
    const socket = {readyState: 1, bufferedAmount: 23000, send: packet => sent.push(packet)};
    sendPCM(socket, new ArrayBuffer(640), 24000);
    socket.bufferedAmount = 23800;
    assert.throws(() => sendPCM(socket, new ArrayBuffer(640), 24000), /Tarmoq/);
    assert.equal(sent.length, 1);
    socket.readyState = 3;
    assert.throws(() => sendPCM(socket, new ArrayBuffer(640), 24000), /uzildi/);
    assert.equal(sent.length, 1);
});

test('worklet capture uses actual hardware rate and flushes remaining samples before its acknowledgement', () => {
    const messages = [];
    let Processor;
    const context = {
        Float32Array, globalThis: {LiveAudio: require('../app/static/js/live_audio.js')}, sampleRate: 44100,
        AudioWorkletProcessor: class { constructor() { this.port = {postMessage: event => messages.push(event)}; } },
        registerProcessor: (_, implementation) => { Processor = implementation; },
    };
    const source = fs.readFileSync(require.resolve('../app/static/js/pcm-worklet.js'), 'utf8').replace(/^import .*;\s*/u, '');
    vm.runInNewContext(source, context);
    const processor = new Processor({processorOptions: {sampleRate: 16000}});
    const output = new Float32Array(128).fill(1);
    for (let i = 0; i < 19; i++) processor.process([[new Float32Array(128).fill(0.5)]], [[output]]);
    processor.port.onmessage({data: {type: 'flush'}});
    assert.equal(messages.at(-1).type, 'flushed');
    const bytes = messages.filter(event => event.type === 'audio').reduce((sum, event) => sum + event.pcm.byteLength, 0);
    assert.equal(bytes / 2, Math.ceil(19 * 128 * 16000 / 44100));
    assert.ok(output.every(sample => sample === 0), 'microphone must never feed speakers');
    assert.equal(processor.process([[]], [[output]]), false);
});

test('transport waits for server ready and sends the final PCM before stop; receives final transcript before cleanup', async () => {
    const original = Object.fromEntries(['AudioContext', 'AudioWorkletNode', 'WebSocket', 'navigator', 'location'].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
    const messages = [], events = [], sockets = [];
    let trackStopped = false;
    const node = () => ({connect() {}, disconnect() {}});
    class Context {
        constructor() { this.state = 'running'; this.sampleRate = 44100; this.destination = {}; this.audioWorklet = {addModule: async () => {}}; }
        createMediaStreamSource() { return node(); }
        createGain() { return {...node(), gain: {value: 1}}; }
        close() { this.state = 'closed'; return Promise.resolve(); }
    }
    class Worklet {
        constructor() {
            this.port = {
                postMessage: () => {
                    this.port.onmessage({data: {type: 'audio', pcm: encodePCM16(Float32Array.of(0.5))}});
                    this.port.onmessage({data: {type: 'flushed'}});
                }, close() {},
            };
        }
        connect() {}
        disconnect() {}
    }
    class Socket {
        constructor(url) { this.url = url; this.readyState = 1; this.bufferedAmount = 0; sockets.push(this); }
        send(message) {
            messages.push(message);
            if (typeof message === 'string' && JSON.parse(message).type === 'stop') {
                this.onmessage({data: JSON.stringify({type: 'conversation.item.input_audio_transcription.completed', item_id: 'tail', transcript: 'buyurtma'})});
                this.onmessage({data: JSON.stringify({type: 'stopped'})});
            }
        }
        close() { this.readyState = 3; }
    }
    try {
        Object.assign(globalThis, {AudioContext: Context, AudioWorkletNode: Worklet, WebSocket: Socket, location: {protocol: 'https:', host: 'test.example'}});
        Object.defineProperty(globalThis, 'navigator', {configurable: true, value: {mediaDevices: {getUserMedia: async () => ({getTracks: () => [{stop() { trackStopped = true; }}]})}}});
        const capture = new MicrophoneStream({language: 'ru', agent: false, onEvent: event => events.push(event)});
        const starting = capture.start();
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(capture.state, 'starting');
        assert.equal(messages.length, 0);
        assert.match(sockets[0].url, /language=ru/);
        assert.match(sockets[0].url, /agent=false/);
        sockets[0].onmessage({data: JSON.stringify({type: 'ready', sample_rate: 16000, provider: 'local'})});
        await starting;
        assert.equal(capture.state, 'running');
        assert.equal(await capture.stop(), true);
        assert.ok(messages[0] instanceof ArrayBuffer, 'flush PCM precedes stop control');
        assert.equal(JSON.parse(messages[1]).type, 'stop');
        assert.equal(events.find(event => event.type.endsWith('.completed')).transcript, 'buyurtma');
        assert.equal(capture.state, 'closed');
        assert.equal(trackStopped, true);
        assert.equal(sockets[0].readyState, 3);
    } finally {
        for (const [key, descriptor] of Object.entries(original)) {
            if (descriptor) Object.defineProperty(globalThis, key, descriptor);
            else delete globalThis[key];
        }
    }
});

function callPageHarness() {
    const requests = [], elements = new Map();
    const drawing = new Proxy({}, {get: (_, key) => key === 'createRadialGradient' ? () => ({addColorStop() {}}) : () => {}});
    const element = () => ({
        style: {}, value: '', textContent: '', className: '', innerHTML: '', children: [],
        addEventListener() {}, setAttribute() {}, getContext: () => drawing,
        appendChild(child) { this.children.push(child); },
    });
    const document = {
        getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
        createElement: element, querySelectorAll: () => [], body: element(),
    };
    const context = vm.createContext({
        document, window: {addEventListener() {}}, performance, console, Uint8Array,
        AbortController, URLSearchParams, setTimeout, clearTimeout, requestAnimationFrame() {},
        fetch(url, options) {
            return new Promise(resolve => requests.push({url, options, resolve}));
        },
    });
    const page = fs.readFileSync(require.resolve('../app/templates/elevenlabs_call.html'), 'utf8');
    const script = [...page.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/gu)].map(match => match[1]).join('\n');
    vm.runInContext(script, context);
    vm.runInContext("isCallActive = true; langSelect.value = 'uz-UZ';", context);
    return {context, requests, elements, run: source => vm.runInContext(source, context)};
}

test('/call partials never trigger AI, final duplicates commit once and low confidence is forwarded', async () => {
    const harness = callPageHarness();
    harness.run("handleLiveEvent({type:'transcript.partial',item_id:'one',transcript:'buyur'});");
    harness.run("handleLiveEvent({type:'transcript.partial',item_id:'one',transcript:'buyurtma'});");
    assert.equal(harness.requests.length, 0);
    assert.equal(harness.elements.get('liveSubtitle').textContent, 'Siz: buyurtma …');
    harness.run("handleLiveEvent({type:'conversation.item.input_audio_transcription.completed',item_id:'one',transcript:'buyurtma',needs_confirmation:true});");
    harness.run("handleLiveEvent({type:'conversation.item.input_audio_transcription.completed',item_id:'one',transcript:'buyurtma'});");
    assert.equal(harness.requests.length, 1);
    assert.equal(JSON.parse(harness.requests[0].options.body).needs_confirmation, true);
    assert.equal(harness.elements.get('chatLog').children.filter(child => child.className === 'msg user').length, 1);
    harness.run('cancelEdgeTurn();');
});

test('/call barge-in aborts current request and ignores a late response even if the HTTP server completes it', async () => {
    const harness = callPageHarness();
    harness.run("handleEdgeTurboTurn('buyurtmam qayerda');");
    assert.equal(harness.requests.length, 1);
    harness.run("handleLiveEvent({type:'input_audio_buffer.speech_started',item_id:'two'});");
    assert.equal(harness.requests[0].options.signal.aborted, true);
    harness.requests[0].resolve({ok: true, json: async () => ({reply: 'stale answer'})});
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(harness.elements.get('chatLog').children.filter(child => child.className === 'msg ai').length, 0);
    assert.equal(harness.run('currentState'), 'listening');
});

test('/call superseded utterances remain visible and the next response uses combined text without duplicate history', () => {
    const harness = callPageHarness();
    harness.run("handleLiveEvent({type:'conversation.item.input_audio_transcription.completed',item_id:'one',transcript:'buyurtma',superseded:true});");
    assert.equal(harness.requests.length, 0);
    assert.equal(harness.run('conversationHistory[0].content'), 'buyurtma');
    harness.run("handleLiveEvent({type:'conversation.item.input_audio_transcription.completed',item_id:'two',transcript:'raqami 123',agent_text:'buyurtma raqami 123'});");
    const payload = JSON.parse(harness.requests[0].options.body);
    assert.equal(payload.message, 'buyurtma raqami 123');
    assert.deepEqual(payload.history, []);
    assert.equal(harness.run('conversationHistory.length'), 2);
    harness.run('cancelEdgeTurn();');
});

test('/call a silent follow-up does not lose an earlier superseded question', () => {
    const harness = callPageHarness();
    harness.run("handleLiveEvent({type:'conversation.item.input_audio_transcription.completed',item_id:'one',transcript:'buyurtmam qayerda',superseded:true});");
    harness.run("handleLiveEvent({type:'conversation.item.input_audio_transcription.completed',item_id:'two',transcript:'',agent_text:'buyurtmam qayerda'});");
    assert.equal(harness.requests.length, 1);
    assert.equal(JSON.parse(harness.requests[0].options.body).message, 'buyurtmam qayerda');
    assert.equal(harness.run('conversationHistory.length'), 1);
    harness.run('cancelEdgeTurn();');
});

function livePageHarness() {
    const elements = new Map(), sent = [];
    const context = vm.createContext({
        document: {getElementById(id) {
            if (!elements.has(id)) elements.set(id, {
                textContent: '', disabled: false, pause() {}, load() {}, removeAttribute() {},
                play: async () => {},
            });
            return elements.get(id);
        }},
        window: {addEventListener() {}}, console, sent,
    });
    const page = fs.readFileSync(require.resolve('../app/templates/live_stt.html'), 'utf8');
    const script = [...page.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/gu)].map(match => match[1]).join('\n');
    vm.runInContext(script, context);
    vm.runInContext('capture = {send: event => sent.push(event)}; isRunning = true;', context);
    return {sent, elements, run: source => vm.runInContext(source, context)};
}

test('/live-stt acknowledges assistant history only after every audio chunk actually ends, once', () => {
    const harness = livePageHarness();
    harness.run("onEvent({type:'agent_thinking',turn_id:'one'});");
    harness.run("onEvent({type:'agent_audio',turn_id:'one',index:0,audio:'first'});");
    harness.run("onEvent({type:'agent_audio',turn_id:'one',index:1,audio:'second'});");
    harness.run("onEvent({type:'agent_done',turn_id:'one',chunks:2});");
    assert.equal(harness.sent.length, 0, 'generation completion is not playback completion');
    harness.elements.get('agent-player').onended();
    assert.equal(harness.sent.length, 0, 'second chunk has not played');
    harness.elements.get('agent-player').onended();
    assert.equal(harness.sent.length, 1);
    assert.equal(harness.sent[0].type, 'agent_playback_done');
    assert.equal(harness.sent[0].turn_id, 'one');
    harness.run("onEvent({type:'agent_done',turn_id:'one',chunks:2});");
    assert.equal(harness.sent.length, 1);
});

test('/live-stt never acknowledges failed, skipped or interrupted assistant audio as heard', () => {
    for (const scenario of ['failed', 'skipped', 'interrupted']) {
        const harness = livePageHarness();
        harness.run("onEvent({type:'agent_thinking',turn_id:'one'});");
        if (scenario === 'skipped') {
            harness.run("onEvent({type:'agent_audio',turn_id:'one',index:1,audio:'second'});");
            harness.run("onEvent({type:'agent_done',turn_id:'one',chunks:2});");
            harness.elements.get('agent-player').onended();
        } else {
            harness.run("onEvent({type:'agent_audio',turn_id:'one',index:0,audio:'first'});");
            harness.run("onEvent({type:'agent_done',turn_id:'one',chunks:1});");
            if (scenario === 'failed') harness.elements.get('agent-player').onerror();
            else harness.run("onEvent({type:'input_audio_buffer.speech_started',item_id:'two'});");
        }
        assert.equal(harness.sent.filter(event => event.type === 'agent_playback_done').length, 0, scenario);
    }
});
