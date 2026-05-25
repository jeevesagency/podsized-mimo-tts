FROM node:20-alpine

# ffmpeg is used for dynamic atempo (post-process tempo adjustment of PCM)
# to land each summary at ~16 chars/sec regardless of voice variation.
RUN apk add --no-cache ffmpeg

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --omit=dev

COPY src ./src

ENV NODE_ENV=production
ENV PORT=3000
EXPOSE 3000

CMD ["node", "src/index.js"]
