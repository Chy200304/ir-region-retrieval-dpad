# Data and model provenance

- Dataset: COCO128, obtained from [LibreYOLO/coco128](https://huggingface.co/datasets/LibreYOLO/coco128); images and box annotations derive from [Microsoft COCO](https://cocodataset.org/).
- Archive SHA256: `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e`.
- Original full images and cropped thumbnails are not committed to this public repository. `prepare_data.py` downloads and verifies the source archive, then generates local thumbnails from `metadata.json`.
- The source archive's GPL v3 notice is retained in `data/UPSTREAM_LICENSE.txt`. Original photographs retain their source-specific rights; the archive notice does not establish a uniform license for every photograph. See [COCO terms](https://cocodataset.org/#termsofuse).
- Every region is traceable to its image ID and pixel crop in `metadata.json`; `image_sources.csv` provides COCO explore links.
- Model: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32), revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Model weights are not committed.
- An initial automatic safetensors conversion at revision `c237dc49a33fc61debc9276459120b7eac67e7ef` was checked tensor-by-tensor against the original PyTorch model: all 400 tensors were equal. Future builds pin the original revision and weight format.
- Cached features are 512-dimensional L2-normalized vectors from a frozen model, not a newly trained model.
- This project is inspired by [DPAD](https://github.com/mrazhou/DPAD); it does not copy that repository's training implementation or claim to reproduce its segmentation results.
- Project results in `results/` come from actual local execution. The failed score variants are reported as negative results.

Access date: 2026-11-06.
