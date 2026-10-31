/* Shared microphone transport for /call and /live-stt. No provider keys in the browser. */
(function (root, factory) {
    const api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    root.LiveAudio = api;
})(globalThis, function () {
    'use strict';

    class StreamingResampler {
        constructor(inputRate, outputRate) {
            if (!(inputRate > 0 && outputRate > 0)) throw new RangeError('Invalid sample rate');
            this.step = inputRate / outputRate;
            this.position = 0;
            this.tail = new Float32Array(0);
            this.inputCount = 0;
            this.outputCount = 0;
            this.lastSample = 0;
        }

        process(samples) {
            if (!samples.length) return new Float32Array(0);
            this.inputCount += samples.length;
            this.lastSample = samples[samples.length - 1];
            const all = new Float32Array(this.tail.length + samples.length);
            all.set(this.tail);
            all.set(samples, this.tail.length);
            const result = [];
            // Both the fractional phase AND the unused samples survive callback boundaries.
            while (this.position + 1 < all.length) {
                const index = Math.floor(this.position);
                const fraction = this.position - index;
                result.push(all[index] + (all[index + 1] - all[index]) * fraction);
                this.position += this.step;
            }
            const consumed = Math.min(Math.floor(this.position), all.length);
            this.tail = all.slice(consumed);
            this.position -= consumed;
            this.outputCount += result.length;
            return Float32Array.from(result);
        }

        flush() {
            // Finish only the remaining real duration; never manufacture a silence tail.
            const remaining = Math.max(0, Math.ceil(this.inputCount / this.step - 1e-9) - this.outputCount);
            const result = new Float32Array(remaining);
            result.fill(this.lastSample);
            this.outputCount += remaining;
            this.tail = new Float32Array(0);
            return result;
        }
    }

    function encodePCM16(samples) {
        const buffer = new ArrayBuffer(samples.length * 2);
        const view = new DataView(buffer);
        for (let i = 0; i < samples.length; i++) {
            const value = Number.isFinite(samples[i]) ? Math.max(-1, Math.min(1, samples[i])) : 0;
            view.setInt16(i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
        }
        return buffer;
    }

    class PCMPacketizer {
        constructor(packetSamples, emit) {
            this.size = Math.round(packetSamples);
            if (this.size < 1) throw new RangeError('Invalid packet size');
            this.emit = emit;
            this.buffer = new Float32Array(this.size);
            this.used = 0;
        }

        push(samples) {
            let offset = 0;
            while (offset < samples.length) {
                const count = Math.min(this.size - this.used, samples.length - offset);
                this.buffer.set(samples.subarray(offset, offset + count), this.used);
                this.used += count;
                offset += count;
                if (this.used === this.size) this.flush();
            }
        }

        flush() {
            if (!this.used) return;
            this.emit(encodePCM16(this.buffer.subarray(0, this.used)));
            this.used = 0;
        }
    }

    function sendPCM(socket, packet, maxBufferedBytes) {
        if (socket.readyState !== 1) throw new Error('Mikrofon aloqasi uzildi. Qayta ulang.');
        if (!(packet instanceof ArrayBuffer) || packet.byteLength % 2) throw new Error('Invalid PCM packet');
        if (socket.bufferedAmount + packet.byteLength > maxBufferedBytes) {
            throw new Error('Tarmoq ovoz oqimiga ulgurmayapti. Aloqani tekshirib, qayta ulang.');
        }
        socket.send(packet);
    }

    class MicrophoneStream {
        constructor(options = {}) {
            this.options = options;
            this.state = 'idle';
            this.socket = null;
            this.stream = null;
            this.context = null;
            this.source = null;
            this.node = null;
            this.sink = null;
            this.inputFilters = [];
            this.error = null;
            this.stopPromise = null;
        }

        async start() {
            if (this.state !== 'idle') throw new Error('Microphone already started');
            this.state = 'starting';
            try {
                const AudioContextClass = globalThis.AudioContext || globalThis.webkitAudioContext;
                if (!AudioContextClass || !globalThis.navigator?.mediaDevices?.getUserMedia) {
                    throw new Error('Mikrofon uchun HTTPS yoki localhost va zamonaviy brauzer kerak.');
                }
                this.context = this.options.context || new AudioContextClass();
                if (this.context.state === 'suspended') await this.context.resume();
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true},
                    video: false,
                });
                if (this.state !== 'starting') {
                    stream.getTracks().forEach(track => track.stop());
                    throw new Error('Ulanish bekor qilindi.');
                }
                this.stream = stream;
                this.source = this.context.createMediaStreamSource(stream);
                const params = new URLSearchParams({
                    language: this.options.language || 'uz',
                    silence_ms: String(this.options.silenceMs || 600),
                    agent: this.options.agent ? 'true' : 'false',
                });
                this.socket = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/live-stt?' + params);
                this.socket.binaryType = 'arraybuffer';
                const ready = await new Promise((resolve, reject) => {
                    this.rejectReady = reject;
                    const timer = setTimeout(() => reject(new Error('Nutq modeli tayyor bo‘lmadi. Server holatini tekshiring.')), this.options.readyTimeoutMs || 90000);
                    this.readyTimer = timer;
                    this.socket.onmessage = event => {
                        let message;
                        try { message = JSON.parse(event.data); } catch (_) { this.fail(new Error('Serverdan noto‘g‘ri javob.')); return; }
                        if (message.type === 'ready') {
                            clearTimeout(timer);
                            this.rejectReady = null;
                            resolve(message);
                        } else if (message.type === 'stopped') {
                            this.resolveStopped?.(true);
                        } else if (message.type === 'error') {
                            this.fail(new Error(message.message || 'Jonli transkripsiyada xato.'));
                            return;
                        }
                        this.options.onEvent?.(message);
                    };
                    this.socket.onerror = () => this.fail(new Error('Jonli transkripsiyaga ulanish amalga oshmadi.'));
                    this.socket.onclose = () => {
                        if (this.state !== 'stopping' && this.state !== 'closed') this.fail(new Error('Ovoz aloqasi uzildi. Qayta ulang.'));
                        this.resolveStopped?.(false);
                    };
                });
                if (this.state !== 'starting') throw this.error || new Error('Ulanish bekor qilindi.');
                this.sampleRate = Number(ready.sample_rate);
                if (![8000, 16000, 24000, 48000].includes(this.sampleRate)) throw new Error('Server audio formati qo‘llanmaydi.');
                this.maxBufferedBytes = this.sampleRate * 2 * 0.75;
                this.sink = this.context.createGain();
                this.sink.gain.value = 0;
                this.sink.connect(this.context.destination);
                if (this.context.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
                    await this.context.audioWorklet.addModule('/app/static/js/pcm-worklet.js');
                    if (this.state !== 'starting') throw this.error || new Error('Ulanish bekor qilindi.');
                    this.node = new AudioWorkletNode(this.context, 'live-pcm-capture', {
                        numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
                        processorOptions: {sampleRate: this.sampleRate},
                    });
                    this.node.port.onmessage = event => {
                        if (event.data.type === 'audio') this.sendPacket(event.data.pcm);
                        else if (event.data.type === 'flushed') this.resolveFlushed?.();
                    };
                    this.node.onprocessorerror = () => this.fail(new Error('Mikrofonni qayta ulang: audio ishlovi to‘xtadi.'));
                } else {
                    // Older browsers use the same phase-preserving encoder, with a larger callback.
                    this.resampler = new StreamingResampler(this.context.sampleRate, this.sampleRate);
                    this.packetizer = new PCMPacketizer(this.sampleRate / 50, packet => this.sendPacket(packet));
                    this.node = this.context.createScriptProcessor(1024, 1, 1);
                    this.node.onaudioprocess = event => {
                        event.outputBuffer.getChannelData(0).fill(0);
                        if (this.state === 'running') this.packetizer.push(this.resampler.process(event.inputBuffer.getChannelData(0)));
                    };
                }
                this.state = 'running';
                // Remove frequencies above the negotiated ASR bandwidth before downsampling.
                // Cascaded low-pass sections prevent high-frequency microphone noise from aliasing.
                let inputNode = this.source;
                if (this.context.sampleRate > this.sampleRate && this.context.createBiquadFilter) {
                    for (let index = 0; index < 2; index++) {
                        const filter = this.context.createBiquadFilter();
                        filter.type = 'lowpass';
                        filter.frequency.value = this.sampleRate * 0.45;
                        filter.Q.value = Math.SQRT1_2;
                        inputNode.connect(filter);
                        this.inputFilters.push(filter);
                        inputNode = filter;
                    }
                }
                inputNode.connect(this.node);
                this.node.connect(this.sink);
                return ready;
            } catch (error) {
                if (this.state !== 'closed') this.fail(error);
                throw error;
            } finally {
                clearTimeout(this.readyTimer);
                this.rejectReady = null;
            }
        }

        sendPacket(packet) {
            if (!['running', 'stopping'].includes(this.state)) return;
            try { sendPCM(this.socket, packet, this.maxBufferedBytes); }
            catch (error) { this.fail(error); }
        }

        send(event) {
            if (this.socket?.readyState === 1) this.socket.send(JSON.stringify(event));
        }

        fail(error) {
            if (this.error || this.state === 'closed') return;
            this.error = error;
            this.rejectReady?.(error);
            this.resolveStopped?.(false);
            this.release();
            this.options.onError?.(error);
        }

        stop() {
            if (!this.stopPromise) this.stopPromise = this.finish();
            return this.stopPromise;
        }

        async finish() {
            if (this.state !== 'running') {
                this.rejectReady?.(new Error('Ulanish bekor qilindi.'));
                this.release();
                return false;
            }
            this.state = 'stopping';
            // Stop input first; flush the worklet's last PCM packet before sending the stop marker.
            try { this.source.disconnect(); } catch (_) {}
            if (this.node?.port) {
                const audioFlushed = await new Promise(resolve => {
                    const timer = setTimeout(() => resolve(false), 1500);
                    this.resolveFlushed = () => { clearTimeout(timer); resolve(true); };
                    this.node.port.postMessage({type: 'flush'});
                });
                if (!audioFlushed) {
                    this.fail(new Error('Mikrofonning oxirgi audio qismi yuborilmadi. Yakuniy matnni tekshiring.'));
                    return false;
                }
            } else if (this.packetizer) {
                this.packetizer.push(this.resampler.flush());
                this.packetizer.flush();
            }
            this.stream?.getTracks().forEach(track => track.stop());
            let flushed = false;
            if (this.socket?.readyState === 1 && !this.error) {
                flushed = await new Promise(resolve => {
                    const timer = setTimeout(() => resolve(false), this.options.stopTimeoutMs || 35000);
                    this.resolveStopped = result => { clearTimeout(timer); resolve(result); };
                    this.send({type: 'stop'});
                });
            }
            this.release();
            if (!flushed && !this.error) this.options.onError?.(new Error('Oxirgi matn tasdiqlanmadi. Saqlangan matnni tekshiring.'));
            return flushed;
        }

        release() {
            this.state = 'closed';
            clearTimeout(this.readyTimer);
            this.stream?.getTracks().forEach(track => track.stop());
            for (const node of [this.source, ...this.inputFilters, this.node, this.sink]) {
                if (node) { try { node.disconnect(); } catch (_) {} }
            }
            if (this.node?.port) this.node.port.close();
            if (this.node && 'onaudioprocess' in this.node) this.node.onaudioprocess = null;
            if (this.socket) {
                this.socket.onmessage = null;
                this.socket.onerror = null;
                this.socket.onclose = null;
                try { this.socket.close(); } catch (_) {}
            }
            if (!this.options.context && this.context?.state !== 'closed') this.context?.close().catch(() => {});
        }
    }

    return {StreamingResampler, PCMPacketizer, encodePCM16, sendPCM, MicrophoneStream};
});
