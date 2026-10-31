import './live_audio.js';

class LivePCMCapture extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const targetRate = options.processorOptions.sampleRate;
        this.resampler = new globalThis.LiveAudio.StreamingResampler(sampleRate, targetRate);
        this.packetizer = new globalThis.LiveAudio.PCMPacketizer(targetRate / 50, pcm => {
            this.port.postMessage({type: 'audio', pcm}, [pcm]);
        });
        this.running = true;
        this.port.onmessage = event => {
            if (event.data.type !== 'flush' || !this.running) return;
            this.running = false;
            this.packetizer.push(this.resampler.flush());
            this.packetizer.flush();
            this.port.postMessage({type: 'flushed'});
        };
    }

    process(inputs, outputs) {
        // Silence the monitoring output: audio is sent to the server, never to the speakers.
        for (const output of outputs) for (const channel of output) channel.fill(0);
        const samples = inputs[0]?.[0];
        if (this.running && samples?.length) this.packetizer.push(this.resampler.process(samples));
        return this.running;
    }
}

registerProcessor('live-pcm-capture', LivePCMCapture);
