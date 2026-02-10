THIS IS A LARGE FEATURE BUILD (not a small fix).

CODEX EXECUTION PERMISSIONS (IMPORTANT)
- You are explicitly authorized to make any code changes required to fully achieve the goal, including:
  - adding new Python modules/packages (via requirements or bootstrap installer)
  - adding new folders/files under app/ and assets/
  - refactoring existing endpoints, job runners, and UI JS
  - adding background job execution / event streaming plumbing
  - adding download/extract utilities for Blender + assets
  - adding manifest / caching / pruning subsystems
- If you believe any platform/tool restriction might block a needed change, DO NOT partially implement.
  Instead:
  1) implement the closest safe equivalent that still meets the NON-NEGOTIABLE REQUIREMENTS, and
  2) clearly log a single actionable status message describing what restriction blocked what step.
- You must not stop at “small diffs” if the feature requires broader changes.
  Implement the full pipeline end-to-end as specified, even if it requires substantial modifications.
- You are allowed to modify any file in the repository except:
  - do not remove existing endpoints (keep /jobs/anime-episode-60s-3d intact)
  - do not remove existing env vars; only add defaults when unset as specified

