// Minimal Ogg-Opus packager (RFC 7845). Wraps raw opus packets in an Ogg container.

const OGG_CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
        let r = i << 24;
        for (let j = 0; j < 8; j++) {
            r = (r & 0x80000000) ? ((r << 1) ^ 0x04C11DB7) : (r << 1);
        }
        table[i] = r >>> 0;
    }
    return table;
})();

function oggCrc32(data) {
    let crc = 0;
    for (let i = 0; i < data.length; i++) {
        crc = ((crc << 8) ^ OGG_CRC_TABLE[((crc >>> 24) ^ data[i]) & 0xff]) >>> 0;
    }
    return crc >>> 0;
}

function buildOpusHead({ channels, preSkip, inputSampleRate }) {
    const buf = new Uint8Array(19);
    const view = new DataView(buf.buffer);
    buf.set([0x4f, 0x70, 0x75, 0x73, 0x48, 0x65, 0x61, 0x64], 0);
    buf[8] = 1;
    buf[9] = channels;
    view.setUint16(10, preSkip, true);
    view.setUint32(12, inputSampleRate, true);
    view.setInt16(16, 0, true);
    buf[18] = 0;
    return buf;
}

function buildOpusTags() {
    const vendor = "podsized-mimo-vps";
    const vendorBytes = new TextEncoder().encode(vendor);
    const buf = new Uint8Array(8 + 4 + vendorBytes.length + 4);
    buf.set([0x4f, 0x70, 0x75, 0x73, 0x54, 0x61, 0x67, 0x73], 0);
    const view = new DataView(buf.buffer);
    view.setUint32(8, vendorBytes.length, true);
    buf.set(vendorBytes, 12);
    view.setUint32(12 + vendorBytes.length, 0, true);
    return buf;
}

function buildOggPage({ headerType, granulePos, serialNumber, pageSequence, packets }) {
    const segments = [];
    for (const pkt of packets) {
        let len = pkt.length;
        while (len >= 255) { segments.push(255); len -= 255; }
        segments.push(len);
    }
    if (segments.length > 255) throw new Error(`segment_table overflow (${segments.length})`);
    const payloadSize = packets.reduce((s, p) => s + p.length, 0);
    const headerSize = 27 + segments.length;
    const page = new Uint8Array(headerSize + payloadSize);
    const view = new DataView(page.buffer);
    page.set([0x4f, 0x67, 0x67, 0x53], 0);
    page[4] = 0;
    page[5] = headerType;
    view.setBigUint64(6, granulePos, true);
    view.setUint32(14, serialNumber, true);
    view.setUint32(18, pageSequence, true);
    view.setUint32(22, 0, true);
    page[26] = segments.length;
    for (let i = 0; i < segments.length; i++) page[27 + i] = segments[i];
    let off = headerSize;
    for (const p of packets) { page.set(p, off); off += p.length; }
    view.setUint32(22, oggCrc32(page), true);
    return page;
}

export function packOggOpus(packets, { inputSampleRate, channels, preSkip = 0, frameSizeMs = 20, packetsPerPage = 50 }) {
    const samplesPer48kFrame = (48000 * frameSizeMs) / 1000;
    const serial = Math.floor(Math.random() * 0xffffffff) >>> 0;
    const pages = [];
    pages.push(buildOggPage({ headerType: 0x02, granulePos: 0n, serialNumber: serial, pageSequence: 0, packets: [buildOpusHead({ channels, preSkip, inputSampleRate })] }));
    pages.push(buildOggPage({ headerType: 0x00, granulePos: 0n, serialNumber: serial, pageSequence: 1, packets: [buildOpusTags()] }));
    let pageSeq = 2;
    let granule = 0n;
    for (let i = 0; i < packets.length; i += packetsPerPage) {
        const batch = packets.slice(i, i + packetsPerPage);
        const isLast = i + packetsPerPage >= packets.length;
        granule += BigInt(batch.length) * BigInt(samplesPer48kFrame);
        pages.push(buildOggPage({ headerType: isLast ? 0x04 : 0x00, granulePos: granule, serialNumber: serial, pageSequence: pageSeq++, packets: batch }));
    }
    const total = pages.reduce((s, p) => s + p.length, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const p of pages) { out.set(p, off); off += p.length; }
    return out;
}
