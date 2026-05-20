// MiMo TTS service for self-hosted VPS (Docker via easypanel).
// Receives {text, episode_identifier} from Supabase edge fn, calls MiMo, transcodes WAV→opus,
// uploads to R2 via S3 v4, returns JSON.

import http from 'node:http';
import { Buffer } from 'node:buffer';
import { createHmac, createHash } from 'node:crypto';
import OpusScript from 'opusscript';
import { packOggOpus } from './ogg_opus.js';

const {
    PORT = '3000',
    WORKER_SHARED_SECRET,
    DEEPINFRA_API_KEY,
    R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY,
    R2_ENDPOINT,
    R2_BUCKET,
    R2_PUBLIC_URL,
} = process.env;

for (const k of ['WORKER_SHARED_SECRET', 'DEEPINFRA_API_KEY', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_ENDPOINT', 'R2_BUCKET', 'R2_PUBLIC_URL']) {
    if (!process.env[k]) {
        console.error(`[FATAL] missing env: ${k}`);
        process.exit(1);
    }
}

const MIMO_ENDPOINT = 'https://api.deepinfra.com/v1/inference/XiaomiMiMo/MiMo-V2.5-tts-voicedesign';

const MIMO_FEMALE = [
    { slug: 'f01_bright_us_late20s',     description: 'Bright, energetic late-twenties American female podcast host. Smiling delivery, quick conversational pacing, warm but punchy.' },
    { slug: 'f02_sharp_anchor_early30s', description: 'Sharp, polished female broadcast news anchor in her early thirties. Crisp articulation, confident, punchy delivery with clear emphasis.' },
    { slug: 'f03_genz_vlogger_early20s', description: 'Casual Gen-Z female podcast host in her early twenties. Modern vlogger-style cadence, warm and friendly, lightly informal, snappy pace.' },
    { slug: 'f04_raspy_pod_mid30s',      description: 'Mid-thirties female podcast host with a warm but lively tone and a slight raspy edge. Conversational, never sleepy, engaged pacing.' },
    { slug: 'f05_witty_commentator_late20s', description: 'Witty late-twenties female commentator. Playful intelligence, dry humor, fast and clean delivery.' },
    { slug: 'f06_british_pod_early30s',  description: 'Bright, articulate British female podcast host in her early thirties. Energetic, warm, with crisp consonants and brisk pacing.' },
    { slug: 'f07_tech_millennial_30s',   description: 'Confident millennial female tech-podcast host. Friendly authority, engaged conversational pace, slightly higher pitch.' },
    { slug: 'f08_latina_us_late20s',     description: 'Lively American female podcast host in her late twenties with a light Latina accent. Warm, animated, conversational, expressive.' },
    { slug: 'f09_radio_host_late20s',    description: 'Polished late-twenties female radio host. Bright, melodic, snappy delivery with a smile in her voice.' },
    { slug: 'f10_sardonic_cohost_early30s', description: 'Sharp, sardonic female late-night co-host in her early thirties. Fast, dry, charismatic, slightly lower-pitched for a female voice.' },
];

const MIMO_MALE = [
    { slug: 'm01_morning_us_late20s',    description: 'Energetic, charismatic American male morning-show podcast host in his late twenties. Smiling delivery, brisk pace, bright tone.' },
    { slug: 'm02_sharp_anchor_mid30s',   description: 'Sharp male broadcast news anchor in his mid-thirties. Crisp, authoritative, even tempo with clear emphasis. Professional but not stiff.' },
    { slug: 'm03_tech_pod_late20s',      description: 'Conversational late-twenties American male tech podcaster. Relaxed but engaged, mid-pitch, modern, never slow.' },
    { slug: 'm04_witty_commentator_30s', description: 'Witty, articulate American male commentator in his early thirties. Light dry humor, punchy and intelligent.' },
    { slug: 'm05_sports_radio_late20s',  description: 'Bright American male sports-radio host in his late twenties. High energy, fast clean delivery, animated.' },
    { slug: 'm06_british_pod_mid30s',    description: 'Mid-thirties British male podcast host. Warm, conversational, energetic, slightly higher pitch, brisk pacing.' },
    { slug: 'm07_polished_anchor_early30s', description: 'Polished male broadcast anchor in his early thirties. Confident, snappy, professional, mid-low pitch.' },
    { slug: 'm08_genz_pod_early20s',     description: 'Casual Gen-Z American male podcast host in his early twenties. Friendly, modern, lightly informal, engaged pacing.' },
    { slug: 'm09_southern_warm_late20s', description: 'Lively, animated American male podcast host in his late twenties with a light Southern warmth. Never slow, conversational.' },
    { slug: 'm10_engaging_mid30s',       description: 'Engaging American male host in his early thirties with a mid pitch, quick natural pacing, and a smile in the voice.' },
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

function wavToOpus(wav) {
    const { samples, sampleRate, channels } = parseWav(wav);
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

function decodeDataUrl(dataUrl) {
    const marker = 'base64,';
    const idx = dataUrl.indexOf(marker);
    const b64 = idx === -1 ? dataUrl : dataUrl.substring(idx + marker.length);
    return new Uint8Array(Buffer.from(b64, 'base64'));
}

// AWS Sig v4 for R2
function sha256Hex(data) {
    return createHash('sha256').update(data).digest('hex');
}
function hmac(key, msg) {
    return createHmac('sha256', key).update(msg).digest();
}
async function signR2Put(url, body) {
    const u = new URL(url);
    const amzDate = new Date().toISOString().replace(/[:-]|\.\d{3}/g, '');
    const dateStamp = amzDate.slice(0, 8);
    const payloadHash = sha256Hex(body);
    const region = 'auto';
    const ch = `host:${u.hostname}\nx-amz-content-sha256:${payloadHash}\nx-amz-date:${amzDate}\n`;
    const sh = 'host;x-amz-content-sha256;x-amz-date';
    const cr = `PUT\n${u.pathname}\n\n${ch}\n${sh}\n${payloadHash}`;
    const cs = `${dateStamp}/${region}/s3/aws4_request`;
    const sts = `AWS4-HMAC-SHA256\n${amzDate}\n${cs}\n${sha256Hex(cr)}`;
    let k = hmac(`AWS4${R2_SECRET_ACCESS_KEY}`, dateStamp);
    k = hmac(k, region);
    k = hmac(k, 's3');
    k = hmac(k, 'aws4_request');
    const signature = createHmac('sha256', k).update(sts).digest('hex');
    const auth = `AWS4-HMAC-SHA256 Credential=${R2_ACCESS_KEY_ID}/${cs}, SignedHeaders=${sh}, Signature=${signature}`;
    return { auth, amzDate, payloadHash };
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
    const ident = payload?.episode_identifier ?? '';
    if (!text || !ident) return send(res, 400, { error: 'missing text or episode_identifier' });

    const voice = pickVoice();
    console.log(`[MIMO] voice=${voice.slug} chars=${text.length} ident=${ident}`);

    try {
        // 1. MiMo
        const t0 = Date.now();
        const mimoResp = await fetch(MIMO_ENDPOINT, {
            method: 'POST',
            headers: { Authorization: `Bearer ${DEEPINFRA_API_KEY}`, 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, voice: voice.description, output_format: 'wav' }),
        });
        const mimoMs = Date.now() - t0;
        if (!mimoResp.ok) {
            const body = await mimoResp.text();
            return send(res, 502, { error: `mimo ${mimoResp.status}: ${body.slice(0, 300)}`, voice_used: voice.slug });
        }
        const mimoJson = await mimoResp.json();
        if (!mimoJson.audio) return send(res, 502, { error: 'mimo missing audio field' });
        const wav = decodeDataUrl(mimoJson.audio);
        console.log(`[MIMO] mimo ${mimoMs}ms, wav ${wav.byteLength}B`);

        // 2. Transcode
        const t1 = Date.now();
        const opus = wavToOpus(wav);
        const transcodeMs = Date.now() - t1;
        console.log(`[MIMO] transcode ${transcodeMs}ms, opus ${opus.byteLength}B`);

        // 3. Upload to R2
        const audioFileName = `audio-summaries/${ident}.opus`;
        const uploadUrl = `${R2_ENDPOINT}/${R2_BUCKET}/${audioFileName}`;
        const sig = await signR2Put(uploadUrl, opus);
        const t2 = Date.now();
        const put = await fetch(uploadUrl, {
            method: 'PUT',
            headers: {
                'Content-Type': 'audio/ogg',
                'x-amz-content-sha256': sig.payloadHash,
                'x-amz-date': sig.amzDate,
                Authorization: sig.auth,
            },
            body: opus,
        });
        const uploadMs = Date.now() - t2;
        if (!put.ok) {
            const body = await put.text();
            return send(res, 502, { error: `r2 upload failed ${put.status}: ${body.slice(0, 200)}` });
        }
        console.log(`[MIMO] r2 upload ${uploadMs}ms`);

        const audio_url = `${R2_PUBLIC_URL}/${audioFileName}`;
        return send(res, 200, {
            audio_url,
            voice_used: voice.slug,
            character_count: text.length,
            audio_provider: 'mimo',
            generation_cost: mimoJson.inference_status?.cost ?? 0,
            runtime_ms: mimoJson.inference_status?.runtime_ms ?? 0,
            opus_bytes: opus.byteLength,
            content_type: 'audio/ogg',
            mimo_ms: mimoMs,
            transcode_ms: transcodeMs,
            upload_ms: uploadMs,
        });
    } catch (e) {
        console.error('[MIMO] error:', e);
        return send(res, 500, { error: e?.message ?? String(e), voice_used: voice.slug });
    }
});

server.listen(Number(PORT), () => {
    console.log(`mimo-tts-service listening on :${PORT}`);
});
