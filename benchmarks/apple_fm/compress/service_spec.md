# Ingest service: requirements for the rewrite

This document collects what the team agreed for the rewrite of the ingest service. It was assembled from three planning meetings in October and from the incident review of the September outage. Most of the background is here for context; the requirements themselves are marked as such.

## Background

The current ingest service was written four years ago as a quick prototype and has been extended many times since. It accepts uploaded files from the web client and the mobile apps, validates them, stores them, and notifies downstream consumers. Over time the code base accumulated several layers of retry logic, two different queueing libraries and a configuration format that nobody fully understands any more. The team has discussed a rewrite at almost every planning meeting this year, and in October it was finally scheduled.

The September outage lasted 3 hours and 40 minutes. A malformed archive caused the extraction worker to exhaust its memory, and because the worker shared a process with the upload handler, uploads failed for every customer during that time. The incident review found 27 retries per failed upload in the logs, which multiplied the load on the storage cluster by a factor of roughly nine.

People in the meetings spent a long time discussing naming, the colour scheme of the internal dashboard and whether the new service should have a mascot. None of that is decided here and it is not part of the requirements.

## Requirements

The service must run fully offline inside the customer's network. Do not introduce any dependency on a public cloud API, a license server or an external telemetry endpoint.

Uploads larger than 2 GiB must be rejected with HTTP status 413 before any bytes are written to storage.

Extraction must run in a separate process from the upload handler, with a hard memory limit of 1.5 GiB per extraction worker.

Retries are limited to at most 3 per upload, with exponential backoff starting at 500 ms.

The public API must stay backwards compatible with version 2.3 of the current service; the endpoint paths /v2/upload and /v2/status keep their request and response formats.

Never log file contents or file names; log only the upload id, the size in bytes and the outcome.

## Performance targets

On the reference hardware (a 16-core server with 64 GB of memory and NVMe storage) the service should sustain 400 uploads per minute of 25 MB files, with a p99 acknowledgement latency under 250 ms. The current service manages about 140 uploads per minute on the same machine, so the target is close to three times the present throughput.

The team expects most of the gain to come from removing the double queueing and from streaming uploads to storage instead of buffering them in memory. Several people suggested moving to a different programming language, but the decision recorded in the second meeting is to stay with Go.

## Schedule and ownership

The rewrite starts on 3 November 2026. Priya Raman owns the upload path, Tomasz Nowak owns extraction and Lena Fischer owns the migration plan. A first internal release is planned for 15 January 2027, and the old service will be switched off no earlier than 1 March 2027.

The migration plan is still a draft. It will run the old and the new service side by side for at least four weeks, comparing outcomes on a mirrored share of production traffic. There was a long discussion about the exact share, which settled on 10 percent for the first two weeks.

## Open questions

Whether the new service should also take over thumbnail generation was left open. Thumbnail generation currently runs in a separate batch job every 15 minutes, and some people would like it to happen at upload time instead. The team agreed to measure the cost first.

Whether to keep the XML status endpoint, which only one customer still uses, was also left open; the account manager will ask that customer in November.

## Appendix: discussion notes

The rest of this section is a loose transcript of the third meeting. It repeats much of the above in less precise words. People agreed that the outage was painful, that the code is hard to change, and that the rewrite is overdue. Several anecdotes were shared about earlier rewrites at other companies, some successful and some not. One person recalled a rewrite that took twice as long as planned because the team kept adding features; the group agreed to keep the scope tight. Another person suggested writing the new service as a set of small microservices, which was discussed at length and rejected in favour of a single service with separate worker processes. The meeting ended with a reminder to book the rooms for the next planning session and a short discussion of the holiday schedule.
