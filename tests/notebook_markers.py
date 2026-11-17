import dataclasses


@dataclasses.dataclass(frozen=True)
class Skip:
    """A notebook that CI never runs; pytest reports the reason."""

    reason: str


# The markers of the notebooks under docs/release. tests/conftest.py adds them to the nbmake items, which CI selects
# with the same marker expressions as the unit tests, and scripts/run-isolated-nb-tests.sh reads them without pytest
# or pixeltable. A notebook takes the markers of the unit tests for the providers and UDFs it calls, or Skip if CI
# never runs it.
NOTEBOOK_MARKERS: dict[str, tuple[str, ...] | Skip] = {
    # Hugging Face models and datasets
    'computed-columns': ('skip_merge_queue',),
    'data-import-huggingface': ('skip_merge_queue',),
    'doc-chunk-for-rag': ('skip_merge_queue',),
    'embedding-indexes': ('skip_merge_queue',),
    'observability': ('skip_merge_queue',),
    'queries-and-expressions': ('skip_merge_queue',),
    'rag-demo': ('skip_merge_queue',),
    'rag-operations': ('skip_merge_queue',),
    'search-semantic-text': ('skip_merge_queue',),
    'search-similar-images': ('skip_merge_queue',),
    'working-with-hugging-face': ('skip_merge_queue',),
    # Hugging Face models, plus OpenAI transcriptions or vision
    'audio-transcriptions': ('skip_merge_queue', 'expensive'),
    'ten-minute-tour': ('skip_merge_queue', 'expensive'),
    # Other model downloads
    'img-detect-objects': ('skip_merge_queue',),
    'img-visualize-detections': ('skip_merge_queue',),
    'working-with-llama-cpp': ('skip_merge_queue',),
    'working-with-ollama': ('skip_merge_queue', 'expensive'),  # PXT-1246
    # Other services not reliable enough for the merge queue
    'data-export-s3': ('skip_merge_queue',),  # writes to a B2 bucket
    'working-with-anthropic': ('skip_merge_queue',),
    # Paid providers other than OpenAI and Anthropic
    'multimodal_backend': ('skip_merge_queue', 'expensive'),  # Gemini, and a Hugging Face model
    'working-with-bedrock': ('skip_merge_queue', 'expensive'),
    'working-with-deepseek': ('skip_merge_queue', 'expensive'),
    'working-with-fal': ('skip_merge_queue', 'expensive'),  # PXT-1233
    'working-with-fireworks': ('skip_merge_queue', 'expensive'),
    'working-with-gemini': ('skip_merge_queue', 'expensive'),
    'working-with-groq': ('skip_merge_queue', 'expensive'),
    'working-with-mistralai': ('skip_merge_queue', 'expensive'),
    'working-with-nebius': ('skip_merge_queue', 'expensive'),
    'working-with-openrouter': ('skip_merge_queue', 'expensive'),
    'working-with-parallel': ('skip_merge_queue', 'expensive'),
    'working-with-replicate': ('skip_merge_queue', 'expensive'),
    'working-with-together': ('skip_merge_queue', 'expensive'),
    # OpenAI speech, transcriptions, vision and image generation
    'audio-text-to-speech': ('expensive',),
    'img-generate-captions': ('expensive',),
    'vision-batch-analysis': ('expensive',),
    'vision-structured-output': ('expensive',),
    'working-with-openai': ('expensive',),
    # Video generation, and the providers whose unit tests are very_expensive
    'video-generate-ai': ('very_expensive',),
    'working-with-jina': ('very_expensive',),
    'working-with-vlmrun': ('very_expensive',),
    'working-with-vlmrun-gateway': ('very_expensive',),
    'working-with-voyageai': ('very_expensive',),
    # Memory, as in the unit tests of the same models, or run time on ubuntu-large
    'img-detection-vs-segmentation': ('resource_intensive',),  # panoptic DETR
    'img-image-to-image': ('resource_intensive',),  # Stable Diffusion, 10 minutes
    'img-promptable-segmentation': ('resource_intensive',),  # SAM3 video segmentation, 19 minutes
    'object-detection-in-videos': ('resource_intensive',),  # YOLOX on video frames, 6 minutes
    'llm-tool-calling': Skip('needs a separately running MCP server'),
    'working-with-bfl': Skip('PXT-1111: out of credits'),
    'working-with-fabric': Skip('PXT-1113: needs a Microsoft Fabric environment'),
    'working-with-fiftyone': Skip('PXT-1117: Voxel51 is left out of the dev environment for security reasons'),
    'working-with-runwayml': Skip('PXT-1120: broken RunwayML integration'),
    'working-with-tigris': Skip('PXT-1122: hard-coded getpass() calls for the credentials and bucket'),
    'working-with-twelvelabs': Skip('PXT-1119: over the rate limit'),
}
