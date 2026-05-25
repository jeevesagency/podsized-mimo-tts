// MiMo TTS service for self-hosted VPS (Docker via easypanel).
// Receives {text} from Supabase edge fn, calls MiMo, transcodes WAV→opus,
// returns opus bytes inline. The edge fn handles R2 upload using its existing credentials.

import http from 'node:http';
import { Buffer } from 'node:buffer';
import { spawn } from 'node:child_process';
import OpusScript from 'opusscript';
import { packOggOpus } from './ogg_opus.js';

const {
    PORT = '3000',
    WORKER_SHARED_SECRET,
    DEEPINFRA_API_KEY,
} = process.env;

for (const k of ['WORKER_SHARED_SECRET', 'DEEPINFRA_API_KEY']) {
    if (!process.env[k]) {
        console.error(`[FATAL] missing env: ${k}`);
        process.exit(1);
    }
}

// OpenAI-compatible endpoint returns raw WAV bytes (vs native inference which wraps in JSON
// AND truncates voicedesign output to ~60s — confirmed via direct test).
const MIMO_ENDPOINT = 'https://api.deepinfra.com/v1/openai/audio/speech';
const MIMO_MODEL = 'XiaomiMiMo/MiMo-V2.5-tts-voicedesign';

// Voice descriptions intentionally avoid speed-loaded language ("quick", "snappy",
// "brisk", "punchy", "fast", "never slow") because MiMo voicedesign reads those as
// pacing instructions and produces rushed audio. Target a calm, natural broadcast
// cadence ~150 words/min so a 750-word script lands around 5 minutes of audio.
const MIMO_FEMALE = [
    { slug: 'f01_bright_us_late20s',     description: 'Warm late-twenties American female podcast host. Smiling delivery, natural conversational pacing, unhurried and clear, warm and inviting.' },
    { slug: 'f02_sharp_anchor_early30s', description: 'Polished female broadcast news anchor in her early thirties. Crisp articulation, confident, steady measured delivery with clear emphasis and natural pauses.' },
    { slug: 'f03_genz_vlogger_early20s', description: 'Casual Gen-Z female podcast host in her early twenties. Modern vlogger-style cadence, warm and friendly, lightly informal, relaxed natural pace.' },
    { slug: 'f04_raspy_pod_mid30s',      description: 'Mid-thirties female podcast host with a warm tone and a slight raspy edge. Conversational, calm and engaged, unhurried pacing with natural breath.' },
    { slug: 'f05_witty_commentator_late20s', description: 'Witty late-twenties female commentator. Playful intelligence, dry humor, clean and measured delivery with thoughtful pauses.' },
    { slug: 'f06_british_pod_early30s',  description: 'Articulate British female podcast host in her early thirties. Warm and engaged, with crisp consonants and natural, measured pacing.' },
    { slug: 'f07_tech_millennial_30s',   description: 'Confident millennial female tech-podcast host. Friendly authority, calm conversational pace, slightly higher pitch, room to breathe between thoughts.' },
    { slug: 'f08_latina_us_late20s',     description: 'American female podcast host in her late twenties with a light Latina accent. Warm and expressive, conversational and unhurried, with natural rhythm.' },
    { slug: 'f09_radio_host_late20s',    description: 'Polished late-twenties female radio host. Bright, melodic, smooth measured delivery with a smile in her voice and natural phrasing.' },
    { slug: 'f10_sardonic_cohost_early30s', description: 'Sardonic female late-night co-host in her early thirties. Dry, charismatic, deliberate and measured delivery, slightly lower-pitched for a female voice.' },
];

const MIMO_MALE = [
    { slug: 'm01_morning_us_late20s',    description: 'Charismatic American male morning-show podcast host in his late twenties. Smiling delivery, natural relaxed pace, bright tone, room to breathe between sentences.' },
    { slug: 'm02_sharp_anchor_mid30s',   description: 'Male broadcast news anchor in his mid-thirties. Crisp, authoritative, steady measured tempo with clear emphasis. Professional but warm.' },
    { slug: 'm03_tech_pod_late20s',      description: 'Conversational late-twenties American male tech podcaster. Relaxed and engaged, mid-pitch, modern, natural unhurried pace.' },
    { slug: 'm04_witty_commentator_30s', description: 'Witty, articulate American male commentator in his early thirties. Light dry humor, measured and intelligent delivery with thoughtful pauses.' },
    { slug: 'm05_sports_radio_late20s',  description: 'Bright American male sports-radio host in his late twenties. Animated and engaged, clean clear delivery, natural pacing, never rushed.' },
    { slug: 'm06_british_pod_mid30s',    description: 'Mid-thirties British male podcast host. Warm, conversational, engaged, slightly higher pitch, natural measured pacing.' },
    { slug: 'm07_polished_anchor_early30s', description: 'Polished male broadcast anchor in his early thirties. Confident, measured, professional, mid-low pitch, natural broadcast cadence.' },
    { slug: 'm08_genz_pod_early20s',     description: 'Casual Gen-Z American male podcast host in his early twenties. Friendly, modern, lightly informal, calm conversational pacing.' },
    { slug: 'm09_southern_warm_late20s', description: 'American male podcast host in his late twenties with a light Southern warmth. Conversational and unhurried, natural easy pacing, room to breathe.' },
    { slug: 'm10_engaging_mid30s',       description: 'Engaging American male host in his early thirties with a mid pitch, calm natural pacing, and a smile in the voice.' },
];

function pickVoice() {
    const pool = Math.random() < 0.5 ? MIMO_FEMALE : MIMO_MALE;
    return pool[Math.floor(Math.random() * pool.length)];
}

function parseWav(buf) {
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const channels = view.getUint16(22, true);
    const sampleRate = view.getUint32(24, true);
    let off = 12;
    while (off < buf.length) {
        const id = String.fromCharCode(buf[off], buf[off+1], buf[off+2], buf[off+3]);
        const size = view.getUint32(off + 4, true);
        if (id === 'data') {
            const samples = new Int16Array(buf.buffer.slice(buf.byteOffset + off + 8, buf.byteOffset + off + 8 + size));
            return { samples, sampleRate, channels };
        }
        off += 8 + size;
    }
    throw new Error('WAV missing data chunk');
}

// Split text into chunks of ~maxChars at sentence boundaries.
// MiMo voicedesign caps output at ~60s of audio per call (≈ 950 chars at 16 chars/sec).
// We use 800 chars to leave headroom for slower-paced voices.
function splitIntoChunks(text, maxChars = 800) {
    const sentences = text.match(/[^.!?]+[.!?]+(\s+|$)|[^.!?]+$/g) ?? [text];
    const chunks = [];
    let current = '';
    for (const s of sentences) {
        if (current.length + s.length <= maxChars) {
            current += s;
        } else {
            if (current) chunks.push(current.trim());
            // If a single sentence exceeds maxChars (rare), still emit it; MiMo can handle up to ~950
            current = s;
        }
    }
    if (current.trim()) chunks.push(current.trim());
    return chunks;
}

// Concatenate PCM Int16 samples from multiple WAVs into one big Int16Array.
// Inserts INTER_CHUNK_SILENCE_MS of silence between chunks so sentence boundaries
// across chunk splits don't butt up against each other. Without this, the model
// produces audibly rushed transitions where one chunk ends and the next begins.
const INTER_CHUNK_SILENCE_MS = 300;
function concatWavs(wavs) {
    const parsed = wavs.map(parseWav);
    const sampleRate = parsed[0].sampleRate;
    const channels = parsed[0].channels;
    const gapSamples = parsed.length > 1
        ? Math.floor((sampleRate * INTER_CHUNK_SILENCE_MS) / 1000) * channels
        : 0;
    const totalAudio = parsed.reduce((s, p) => s + p.samples.length, 0);
    const totalGap = gapSamples * (parsed.length - 1);
    const combined = new Int16Array(totalAudio + totalGap);
    let off = 0;
    for (let i = 0; i < parsed.length; i++) {
        combined.set(parsed[i].samples, off);
        off += parsed[i].samples.length;
        if (i < parsed.length - 1) off += gapSamples; // leave zeros for silence
    }
    return { samples: combined, sampleRate, channels };
}

// Dynamic atempo: stretch PCM so output lands at TARGET_CPS chars/sec.
// MiMo speaks at ~17-19 chars/sec regardless of voice description — see the
// 2026-05-25 probe results (the `speed` parameter on the OpenAI-compat endpoint
// is silently ignored by the voicedesign model). User-validated sweet spot is
// ~16 cps. We only ever slow down (never speed up); MIN_ATEMPO floor protects
// audio quality (below 0.80 it sounds tape-stretched on consonants).
const TARGET_CPS = 16;
const MIN_ATEMPO = 0.80;
const ATEMPO_DEADBAND = 0.02; // skip ffmpeg if the adjustment is < 2%

function computeAtempo(chars, audioSeconds) {
    const cps = chars / audioSeconds;
    if (cps <= TARGET_CPS) return 1.0; // already at or under target — leave alone
    return Math.max(MIN_ATEMPO, TARGET_CPS / cps);
}

async function applyAtempo(pcmInt16, sampleRate, channels, factor) {
    return new Promise((resolve, reject) => {
        const ff = spawn('ffmpeg', [
            '-hide_banner', '-loglevel', 'error',
            '-f', 's16le', '-ar', String(sampleRate), '-ac', String(channels),
            '-i', 'pipe:0',
            '-filter:a', `atempo=${factor.toFixed(3)}`,
            '-f', 's16le',
            'pipe:1',
        ], { stdio: ['pipe', 'pipe', 'pipe'] });
        const out = [];
        let stderr = '';
        ff.stdout.on('data', c => out.push(c));
        ff.stderr.on('data', c => { stderr += c.toString(); });
        ff.on('error', reject);
        ff.on('close', code => {
            if (code !== 0) return reject(new Error(`ffmpeg exit ${code}: ${stderr.slice(-300)}`));
            const buf = Buffer.concat(out);
            // round down to nearest sample pair if odd byte count slipped through
            const sampleCount = Math.floor(buf.byteLength / 2);
            const arr = new Int16Array(sampleCount);
            for (let i = 0; i < sampleCount; i++) arr[i] = buf.readInt16LE(i * 2);
            resolve(arr);
        });
        ff.stdin.on('error', reject);
        ff.stdin.write(Buffer.from(pcmInt16.buffer, pcmInt16.byteOffset, pcmInt16.byteLength));
        ff.stdin.end();
    });
}

function pcmToOpus(samples, sampleRate, channels) {
    const enc = new OpusScript(sampleRate, channels, OpusScript.Application.AUDIO);
    enc.setBitrate(24000);
    const frameSize = (sampleRate * 20) / 1000;
    const packets = [];
    for (let i = 0; i + frameSize <= samples.length; i += frameSize) {
        const frame = samples.subarray(i, i + frameSize);
        const buf = enc.encode(Buffer.from(frame.buffer, frame.byteOffset, frame.byteLength), frameSize);
        packets.push(new Uint8Array(buf));
    }
    enc.delete();
    return packOggOpus(packets, { inputSampleRate: sampleRate, channels, frameSizeMs: 20 });
}

function wavToOpus(wav) {
    const { samples, sampleRate, channels } = parseWav(wav);
    return pcmToOpus(samples, sampleRate, channels);
}

async function callMimo(text, voiceDescription) {
    const resp = await fetch(MIMO_ENDPOINT, {
        method: 'POST',
        headers: { Authorization: `Bearer ${DEEPINFRA_API_KEY}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: MIMO_MODEL, input: text, voice: voiceDescription, response_format: 'wav' }),
    });
    if (!resp.ok) {
        const body = await resp.text();
        throw new Error(`mimo ${resp.status}: ${body.slice(0, 200)}`);
    }
    return new Uint8Array(await resp.arrayBuffer());
}

function decodeDataUrl(dataUrl) {
    const marker = 'base64,';
    const idx = dataUrl.indexOf(marker);
    const b64 = idx === -1 ? dataUrl : dataUrl.substring(idx + marker.length);
    return new Uint8Array(Buffer.from(b64, 'base64'));
}

async function readJsonBody(req) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        req.on('data', c => chunks.push(c));
        req.on('end', () => {
            try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
            catch (e) { reject(e); }
        });
        req.on('error', reject);
    });
}

function send(res, status, obj) {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(obj));
}

const server = http.createServer(async (req, res) => {
    if (req.method === 'GET' && req.url === '/health') {
        return send(res, 200, { ok: true });
    }
    if (req.method !== 'POST') return send(res, 405, { error: 'POST only' });

    const auth = req.headers['authorization'];
    if (auth !== `Bearer ${WORKER_SHARED_SECRET}`) return send(res, 401, { error: 'unauthorized' });

    let payload;
    try { payload = await readJsonBody(req); }
    catch { return send(res, 400, { error: 'invalid JSON body' }); }
    const text = payload?.text ?? '';
    if (!text) return send(res, 400, { error: 'missing text' });

    const voice = pickVoice();
    console.log(`[MIMO] voice=${voice.slug} chars=${text.length}`);

    try {
        // 1. MiMo voicedesign caps each call at ~60s output (~950 chars). Chunk longer scripts.
        const chunks = splitIntoChunks(text, 800);
        console.log(`[MIMO] split ${text.length} chars into ${chunks.length} chunks (avg ${Math.round(text.length / chunks.length)} chars)`);

        const t0 = Date.now();
        // Parallelize chunk calls. DeepInfra MiMo handles concurrent requests well in our testing.
        const wavs = await Promise.all(chunks.map(async (chunk, i) => {
            const ct = Date.now();
            const wav = await callMimo(chunk, voice.description);
            console.log(`[MIMO] chunk ${i + 1}/${chunks.length} (${chunk.length} chars) → ${wav.byteLength}B in ${Date.now() - ct}ms`);
            return wav;
        }));
        const mimoMs = Date.now() - t0;

        // 2. Concatenate PCM, apply dynamic atempo, transcode to opus
        const t1 = Date.now();
        const { samples, sampleRate, channels } = concatWavs(wavs);
        const rawSeconds = samples.length / sampleRate;
        const rawCps = text.length / rawSeconds;
        const atempo = computeAtempo(text.length, rawSeconds);
        let pcm = samples;
        let atempoMs = 0;
        if (Math.abs(1 - atempo) >= ATEMPO_DEADBAND) {
            const ta = Date.now();
            pcm = await applyAtempo(samples, sampleRate, channels, atempo);
            atempoMs = Date.now() - ta;
        }
        const finalSeconds = pcm.length / sampleRate;
        const finalCps = text.length / finalSeconds;
        const opus = pcmToOpus(pcm, sampleRate, channels);
        const transcodeMs = Date.now() - t1 - atempoMs;
        console.log(`[MIMO] cps ${rawCps.toFixed(2)} → ${finalCps.toFixed(2)} (atempo=${atempo.toFixed(3)}), audio ${rawSeconds.toFixed(1)}s → ${finalSeconds.toFixed(1)}s, mimo ${mimoMs}ms, atempo ${atempoMs}ms, transcode ${transcodeMs}ms, opus ${opus.byteLength}B`);

        // 3. Return opus bytes inline; caller handles R2 upload.
        res.writeHead(200, {
            'Content-Type': 'audio/ogg',
            'X-Voice-Used': voice.slug,
            'X-Character-Count': String(text.length),
            'X-Audio-Provider': 'mimo',
            'X-Generation-Cost': '0',
            'X-Mimo-Ms': String(mimoMs),
            'X-Atempo-Ms': String(atempoMs),
            'X-Atempo-Factor': atempo.toFixed(3),
            'X-Raw-Cps': rawCps.toFixed(2),
            'X-Final-Cps': finalCps.toFixed(2),
            'X-Transcode-Ms': String(transcodeMs),
            'X-Opus-Bytes': String(opus.byteLength),
            'X-Chunks': String(chunks.length),
            'X-Audio-Seconds': finalSeconds.toFixed(1),
            'Content-Length': String(opus.byteLength),
        });
        res.end(Buffer.from(opus));
    } catch (e) {
        console.error('[MIMO] error:', e);
        return send(res, 500, { error: e?.message ?? String(e), voice_used: voice.slug });
    }
});

server.listen(Number(PORT), () => {
    console.log(`mimo-tts-service listening on :${PORT}`);
});
