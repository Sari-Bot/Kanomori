# Kanomori Project White Paper

## A Multimodal Retrieval System for VTuber Livestream Archives

### 1. Executive Summary

Kanomori is a multimodal retrieval platform designed for large-scale VTuber livestream archives. The project originated from a simple personal idea: creating a search engine capable of navigating the vast archive of livestreams, karaoke sessions, conversations, and memorable moments created by **鹿乃 (Kano Mahoro)**. As the concept evolved, it became clear that the same challenges faced when searching Kano's content also apply to many VTuber archives.

Kanomori allows users to locate specific livestreams, moments, songs, conversations, or source clips through incomplete and heterogeneous inputs, including screenshots, transcript fragments, audio snippets, lyrics, vague memories, and metadata.

Unlike conventional video archive platforms that rely mainly on titles, tags, or manually written descriptions, Kanomori focuses on **moment-level retrieval**. The system is intended to answer questions such as:

- “Which stream is this screenshot from?”
- “When did Kano say this sentence?”
- “Which karaoke stream contained this song?”
- “Where is the original source of this edited clip?”
- “I vaguely remember she talked about university while playing Minecraft. Can I find it?”

The core value of the project is not merely video search. It is the ability to transform fragmented fan memory into searchable, verifiable archive evidence.

### 2. Background and Problem Statement

The idea behind Kanomori began with a practical problem: finding specific moments within the extensive archive of 鹿乃 (Kano Mahoro). Like many VTubers, Kano has accumulated hundreds or thousands of hours of content across chatting streams, gaming streams, karaoke streams, special events, collaborations, and other broadcasts.

Existing platforms generally provide only limited retrieval capabilities:

- Search by video title
- Search by channel or date
- Search by manually added tags
- Search by transcript, if available
- Search by short clips, if manually curated

However, viewers often do not remember exact titles, dates, or stream metadata. They usually remember fragments:

- A line of dialogue
- A song lyric
- A visual scene
- A short audio segment
- A meme
- A vague topic
- A screenshot from a clip
- A moment from a long stream

This creates a gap between how users remember content and how current platforms index content.

Kanomori aims to close this gap by building a searchable multimodal memory layer over livestream archives, beginning with Kano's content and potentially expanding to broader VTuber archives.

### 3. Target Users

The primary users include:

1. **VTuber Fans**
   Users who want to rediscover memorable moments, songs, jokes, emotional scenes, or old conversations.
2. **Clip Creators**
   Users who need to find source streams, verify timestamps, collect context, and produce subtitle-ready clips.
3. **Subtitle Groups and Archivists**
   Groups that maintain large archives and need efficient tools for transcript search, audio matching, and source verification.
4. **Researchers and Data Analysts**
   Users who analyze long-term creator behavior, topic evolution, collaboration networks, language use, or community interaction.
5. **Content Moderators and Archive Managers**
   Users responsible for organizing, tagging, and maintaining searchable video collections.

### 4. Product Vision

Kanomori is designed as a **VTuber archive intelligence system**, not merely a video search engine.

The long-term vision is to provide:

- Multimodal search across text, audio, image, and metadata
- Reverse search from clips to original livestreams
- Moment-level timeline navigation
- Karaoke and song history retrieval
- Vague-memory search
- Evidence-based AI question answering
- Automatic discovery of notable moments
- Creator-specific knowledge graphs and topic timelines

The system should allow users to search in the same way they naturally remember content.

Although the project originated from Kano's archive, the underlying vision is to create a reusable framework for searching long-form VTuber content.

### 5. Core Use Cases

#### 5.1 Screenshot to Source Stream

A user uploads a screenshot from a VTuber livestream or edited clip. The system returns:

- Candidate original livestreams
- Candidate timestamps
- Similar frames
- Surrounding transcript
- Confidence score
- Preview segment

This is especially useful for reverse-searching clips and identifying the original source.

#### 5.2 Transcript Fragment Search

A user inputs a sentence or partial phrase. The text may be inaccurate, incomplete, or based on memory.

The system performs:

- Japanese/Chinese/English text normalization
- Full-text search
- Semantic embedding search
- Transcript-window matching
- Timestamp-level result ranking

The result includes the most likely stream and timestamp.

#### 5.3 Audio Snippet Search

A user uploads a short audio segment, such as:

- A karaoke recording
- A clipped dialogue segment
- A meme audio
- A short segment from social media

The system attempts to match the audio against archived livestream audio and returns the source stream and time range.

For karaoke and music-related content, this may become one of the strongest retrieval methods.

#### 5.4 Karaoke Stream Search

A user searches by:

- Song name
- Lyrics
- Audio snippet
- Screenshot with lyrics
- Stream title
- Fuzzy memory

The system returns:

- All streams where the song was performed
- Timestamp of each performance
- Possible repeated performances
- Lyrics context
- Screenshot preview
- Source video link

This feature is particularly valuable for Kano's extensive singing and karaoke content.

#### 5.5 Vague Memory Search

A user inputs a natural-language memory, such as:

> “There was a time she talked about studying abroad while playing Minecraft.”

The system rewrites this vague memory into multiple retrieval queries:

- Topic query: studying abroad
- Context query: Minecraft stream
- Semantic query: university, overseas study, English learning
- Metadata query: game category = Minecraft

It then retrieves candidate moments and ranks them by transcript relevance, metadata consistency, and timeline evidence.

#### 5.6 Clip Reverse Search

A user provides a short clip from Bilibili, YouTube Shorts, Twitter/X, TikTok, or another platform. The system extracts:

- Audio fingerprint
- Key frames
- OCR text
- Transcript fragment
- Visual scene type

It then identifies the original livestream and timestamp.

This feature has strong appeal because it directly solves a common fan need: finding the original source of edited clips.

### 6. Differentiation and Project Highlights

The project’s value should not be concentrated in generic “AI search.” The key differentiators should be explicit and product-oriented.

#### 6.1 Multimodal Retrieval

Users can search with:

- Screenshot
- Audio snippet
- Transcript fragment
- Lyrics
- Stream title
- Fuzzy description
- Clip segment

The system merges these signals into a unified retrieval pipeline.

#### 6.2 VTuber-Specific Optimization

The platform is optimized for common VTuber scenarios:

- Chatting streams
- Gaming streams
- Karaoke streams
- Superchat readings
- Waiting screens
- Collaboration streams
- Announcement streams
- Meme moments
- Emotional moments

This makes it more specialized than a generic video search tool.

#### 6.3 Moment-Level Search

The system does not only return a video. It returns a specific timestamp or time range.

A good result should answer:

- Which stream?
- Which timestamp?
- What was being said?
- What was on screen?
- Why is this result likely correct?

#### 6.4 Clip Source Verification

The system can help verify the original source of edited clips, reducing misinformation, missing context, and incorrect attribution.

#### 6.5 Memory-Oriented Search

The system supports vague and human-like queries, not only exact keywords.

Example:

> “The time she laughed very hard during a horror game.”

This requires semantic search, metadata filtering, and possibly audio emotion signals.

#### 6.6 Karaoke History

The system can build a searchable singing history for each VTuber:

- Songs performed
- Performance dates
- Repeated songs
- Lyrics-based search
- Audio-based matching
- Stream-level song timeline

For Kano's archive, this feature is expected to be one of the most valuable discovery tools.

#### 6.7 Evidence-Based AI Q&A

Instead of answering from speculation, the system can cite source moments:

> “When did she first mention learning guitar?”

The answer should include timestamps, transcript snippets, and source video references.

### 7. Technical Architecture

The recommended architecture is a hybrid system based on mature components, with custom business logic for VTuber-specific retrieval.

#### 7.1 Data Ingestion Layer

Input sources:

- Local video files
- YouTube archive links
- Bilibili archive links
- Existing metadata files
- Manually added stream records

Processing steps:

1. Register video metadata
2. Extract audio
3. Extract frames
4. Generate transcript
5. Generate OCR text
6. Generate visual features
7. Generate audio fingerprints
8. Build indexes
9. Store timestamp-level mappings

#### 7.2 Storage Layer

Recommended storage:

- PostgreSQL for structured metadata
- Local filesystem or object storage for video frames and preview clips
- FAISS or Qdrant for vector search
- Meilisearch, Elasticsearch, or OpenSearch for full-text search
- Optional MinIO for scalable media storage

Core data entities:

- `videos`
- `frames`
- `transcript_segments`
- `ocr_segments`
- `audio_fingerprints`
- `scene_segments`
- `search_results`
- `user_feedback`

#### 7.3 Transcript Processing

ASR should be used to create timestamped transcripts.

Recommended approach:

- Use faster-whisper or equivalent ASR engine
- Segment transcript into 10–30 second windows
- Store original text and normalized text
- Generate text embeddings
- Index both full-text and semantic representations

Transcript search should combine:

- BM25 keyword matching
- Semantic embedding search
- Japanese text normalization
- Sliding time windows
- Metadata reranking

#### 7.4 Image Processing

Image processing should not rely only on CLIP. The recommended image pipeline is:

1. Frame extraction with ffmpeg
2. Perceptual hashing with pHash/dHash
3. OCR extraction for lyrics, titles, captions, and UI text
4. Scene classification with CLIP or a lightweight classifier
5. Optional visual embeddings for auxiliary retrieval
6. Result reranking based on transcript, OCR, and metadata

CLIP should mainly be used for rough scene routing:

- Singing
- Chatting
- Gaming
- Waiting screen
- Superchat reading
- Announcement
- Collaboration

It should not be treated as the main precision retrieval engine.

#### 7.5 Audio Processing

Audio should be used for:

- ASR transcript generation
- Audio fingerprint matching
- Song or performance identification
- Clip reverse search
- Loudness and emotional peak detection

Audio fingerprinting is especially important for karaoke streams and short clip matching.

#### 7.6 Retrieval Layer

The retrieval system should support different input types.

For transcript input:

1. Normalize query
2. Search transcript full-text index
3. Search transcript embedding index
4. Merge candidate segments
5. Rerank using metadata, scene type, OCR, and time context
6. Return timestamp-level results

For screenshot input:

1. Preprocess image
2. Run OCR
3. Compute pHash/dHash
4. Run scene classification
5. Search image index
6. Search OCR text if text exists
7. Merge and rerank candidates

For audio input:

1. Extract audio fingerprint
2. Search fingerprint index
3. If no exact match, run ASR on the snippet
4. Search transcript index
5. Combine with frame and metadata evidence

For multimodal input:

1. Process each modality independently
2. Generate candidate sets
3. Merge candidates by video and timestamp
4. Apply weighted reranking
5. Return unified results

### 8. Ranking Strategy

A practical ranking formula should be dynamic rather than fixed.

For chatting streams:

```
score =
  0.45 * transcript_score
+ 0.20 * OCR_score
+ 0.15 * metadata_score
+ 0.10 * scene_type_score
+ 0.10 * visual_similarity_score
```

For gaming streams:

```
score =
  0.35 * visual_similarity_score
+ 0.25 * transcript_score
+ 0.15 * OCR_score
+ 0.15 * metadata_score
+ 0.10 * scene_type_score
```

For karaoke streams:

```
score =
  0.35 * audio_match_score
+ 0.25 * OCR_or_lyrics_score
+ 0.20 * transcript_score
+ 0.10 * scene_type_score
+ 0.10 * visual_similarity_score
```

This scene-aware ranking model is central to improving retrieval accuracy.

### 9. MVP Scope

The first MVP should avoid overengineering. The goal is to prove retrieval value using a focused archive, with Kano's content serving as the initial dataset.

Recommended MVP scale:

- 50–200 hours of livestream archives
- Initial focus on 鹿乃 (Kano Mahoro)
- Transcript search
- Screenshot search
- OCR support
- Basic scene classification
- Timestamp-level result output
- Simple web interface

MVP features:

1. Import video
2. Extract frames every 5–10 seconds
3. Generate transcript
4. Generate OCR text
5. Classify scene type
6. Search by transcript
7. Search by screenshot
8. Display source stream and timestamp
9. Show nearby transcript
10. Show preview frames

Features to postpone:

- Full-scale clip reverse search
- Large-scale public deployment
- Custom-trained visual model
- Real-time indexing
- Advanced knowledge graph
- Complex recommendation system

### 10. Development Roadmap

#### Phase 1: Proof of Concept

Goal: Validate whether transcript and screenshot retrieval can find correct livestream moments within Kano's archive.

Main tasks:

- Build local video ingestion script
- Extract audio and frames
- Run ASR
- Run OCR
- Store metadata in PostgreSQL
- Build transcript search
- Build simple screenshot search
- Return candidate timestamps

Expected output:

- Internal demo
- Top-k retrieval results
- Basic confidence score
- Manual evaluation dataset

#### Phase 2: Multimodal MVP

Goal: Combine text, image, OCR, and scene classification.

Main tasks:

- Add CLIP-based scene classification
- Add pHash/dHash near-duplicate matching
- Add OCR index
- Add reranking strategy
- Build simple web UI
- Add preview frames and transcript context

Expected output:

- Usable local web application
- Search by screenshot or transcript
- Timestamp-level result display

#### Phase 3: Audio and Karaoke Search

Goal: Support audio snippet and song-based retrieval.

Main tasks:

- Add audio fingerprinting
- Add lyrics-oriented OCR and transcript matching
- Add song metadata extraction
- Build karaoke search page
- Detect repeated song performances

Expected output:

- Search by song name, lyrics, or audio snippet
- Karaoke history per VTuber

#### Phase 4: Clip Reverse Search

Goal: Identify original livestream sources from edited clips.

Main tasks:

- Extract keyframes from uploaded clips
- Extract audio fingerprints
- Extract OCR text
- Generate transcript from clip audio
- Merge multimodal evidence
- Return source stream and timestamp

Expected output:

- Reverse source search demo
- Strong user-facing differentiation

#### Phase 5: Knowledge and Discovery Layer

Goal: Move from retrieval to archive intelligence.

Main tasks:

- Build topic timelines
- Add vague memory search
- Add evidence-based AI Q&A
- Add collaboration graph
- Add notable moment detection

Expected output:

- Advanced search experience
- AI-assisted archive exploration

### 11. Cost Considerations

The project should control cost by using mature open-source components and staged indexing.

#### 11.1 Offline Cost

Major offline costs:

- ASR processing
- OCR processing
- Image embedding
- Audio fingerprinting
- Frame storage

Cost control strategies:

- Process popular videos first
- Use low-frequency frame extraction for static streams
- Increase frame density only in high-motion segments
- Cache all embeddings and OCR results
- Use batch processing
- Separate ingestion from online search

#### 11.2 Online Cost

Online query cost should remain low.

Most online operations are:

- Text search
- Vector lookup
- Hash lookup
- Metadata filtering
- Reranking over limited candidates

GPU should not be required for most online queries, except when processing newly uploaded images or audio snippets.

#### 11.3 Recommended Deployment for MVP

A single-machine MVP is sufficient:

- CPU server
- Optional consumer GPU for offline processing
- PostgreSQL
- FAISS
- FastAPI backend
- Simple frontend

This avoids premature infrastructure complexity.

### 12. Technical Stack Recommendation

Recommended MVP stack:

- Backend: FastAPI
- Database: PostgreSQL
- Video processing: ffmpeg
- ASR: faster-whisper
- OCR: PaddleOCR or EasyOCR
- Image hash: imagehash
- Vector search: FAISS
- Full-text search: Meilisearch or PostgreSQL full-text search
- Frontend: Vue, React, or Next.js
- Media storage: local filesystem initially; MinIO later

Recommended later upgrades:

- Qdrant for service-based vector search
- Elasticsearch/OpenSearch for advanced transcript retrieval
- Celery/RQ for background processing
- MinIO for object storage
- Docker Compose for deployment

### 13. Evaluation Metrics

The system should be evaluated with practical retrieval metrics.

Recommended metrics:

- Top-1 accuracy
- Top-5 accuracy
- Mean reciprocal rank
- Timestamp error range
- Search latency
- Indexing time per hour of video
- OCR hit rate
- ASR quality
- User correction rate
- Successful source identification rate

Example evaluation task:

- Prepare 100 known screenshots
- Prepare 100 transcript queries
- Prepare 50 audio snippets
- Prepare 50 vague-memory queries
- Measure whether the correct stream and timestamp appear in Top-5

For this product, Top-5 accuracy may be more important than Top-1 accuracy because users can visually verify candidate results.

### 14. Risks and Mitigation

#### 14.1 Low-Information Visual Scenes

Risk: Chatting streams may look visually similar for long periods.

Mitigation:

- Use transcript-first retrieval
- Use OCR and metadata
- Reduce reliance on CLIP
- Use scene-aware ranking

#### 14.2 ASR Errors

Risk: Japanese ASR may misrecognize names, jokes, songs, and game terminology.

Mitigation:

- Use fuzzy search
- Use semantic embeddings
- Use domain vocabulary
- Allow manual corrections
- Store confidence scores

#### 14.3 Copyright and Platform Restrictions

Risk: Livestream archives may have copyright, redistribution, or platform terms issues.

Mitigation:

- Store only necessary metadata and derived indexes when possible
- Provide source links instead of redistributing full video
- Limit preview duration
- Support private/local deployment
- Respect takedown requests

#### 14.4 High Processing Cost

Risk: Large archives require significant compute and storage.

Mitigation:

- Incremental indexing
- Popularity-based prioritization
- Adaptive frame sampling
- Batch processing
- Cache all intermediate results

#### 14.5 Ambiguous Results

Risk: Multiple streams may contain similar dialogue or visuals.

Mitigation:

- Return multiple candidates
- Show transcript context
- Show preview frames
- Provide confidence scores
- Allow user feedback to improve ranking

### 15. Business and Community Value

Kanomori can serve as both a fan tool and a creator-supportive archive system.

Potential value:

- Help fans rediscover memorable moments
- Help clip creators find source material
- Help subtitle groups work more efficiently
- Help communities preserve creator history
- Help researchers analyze long-term content trends
- Reduce misinformation from contextless clips
- Create searchable karaoke histories
- Build creator-specific knowledge archives

The product’s strongest community appeal comes from its ability to make long-form livestream memory searchable.

### 16. Recommended Positioning

The project should not be positioned merely as:

> “A VTuber video search engine.”

A stronger positioning is:

> “A multimodal memory search engine for VTuber livestream archives.”

Alternative positioning:

> “Trace.moe for VTuber livestreams, enhanced with transcript, audio, OCR, and fuzzy memory search.”

Or:

> “A source-finding and moment-retrieval system for VTuber archives.”

The key message should emphasize:

- Search by what users remember
- Retrieve exact source moments
- Support screenshots, audio, lyrics, and vague descriptions
- Focus on VTuber-specific livestream scenarios

### 17. Conclusion

Kanomori addresses a real gap in VTuber archive navigation. Current platforms are good at storing videos, but weak at recovering exact moments from incomplete memory. By combining transcript search, OCR, audio fingerprinting, perceptual image matching, scene classification, and multimodal reranking, the system can provide a specialized retrieval experience that generic platforms cannot easily offer.

The recommended development path is incremental:

1. Build transcript search
2. Add screenshot and OCR search
3. Add scene-aware reranking
4. Add audio snippet search
5. Add karaoke and clip reverse search
6. Add vague-memory and AI Q&A features

The project should use mature open-source components rather than developing all algorithms from scratch. Its differentiation should come from vertical integration, VTuber-specific ranking logic, and user-facing workflows.

The core product promise is simple:

> Even if the user only remembers a line, a screenshot, a song, a sound, or a vague moment, V-Moment helps recover the original livestream and timestamp.