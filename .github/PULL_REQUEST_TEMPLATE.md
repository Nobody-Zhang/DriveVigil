<!-- Thanks for contributing! Please fill in the sections below. -->

## Summary

<!-- What does this PR do, and why? -->

## Related issue

<!-- e.g. Closes #123 -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor / cleanup
- [ ] Other (describe):

## Checklist

- [ ] `ruff check .` passes (or only touches non-excluded files)
- [ ] `ruff format --check .` passes
- [ ] `pytest` passes
- [ ] I did **not** edit vendored trees (`**/yolo/models|utils/`, `apig_sdk/`,
      `edge/cloud_finetune/yolov5/`, the SIS SDK, `edge/frontend/`, …)
- [ ] If I changed shared algorithm logic, I mirrored it across the relevant
      near-duplicate variants (`cloud/preliminary` ↔ `cloud/semifinal`,
      `edge/mtcnn` ↔ `edge/ota/mtcnn_landmarks`)
- [ ] No secrets/credentials are committed (everything goes through `.env`)
- [ ] If the change targets `cloud/`, it stays Python 3.7-compatible (ModelArts)

## Notes for reviewers

<!-- Anything else worth knowing: testing done, screenshots, caveats. -->
