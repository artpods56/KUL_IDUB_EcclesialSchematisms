FROM node:26-alpine AS builder

WORKDIR /app/apps/web

COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci

COPY apps/web ./

ARG NEXT_PUBLIC_NOTARIUS_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_NOTARIUS_API_URL=$NEXT_PUBLIC_NOTARIUS_API_URL

RUN npm run build

FROM node:26-alpine AS runner

WORKDIR /app/apps/web

ENV NODE_ENV=production
ENV PORT=3000

COPY --from=builder /app/apps/web ./

EXPOSE 3000

CMD ["npm", "run", "start", "--", "-H", "0.0.0.0"]
