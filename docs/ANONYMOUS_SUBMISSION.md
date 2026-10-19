# Anonymous artifact release

This document describes how to prepare a review artifact for a double-anonymous
submission. It is a release checklist, not a substitute for the target venue's
current call for papers. The submission deadline and link policy of that venue
take precedence.

## Recommended layout

The review artifact should contain only the code, small reproducibility inputs,
and the result summaries that are needed to verify the claims. Keep credentials,
model weights, raw profiler output, cluster scripts, SSH configuration, and the
repository's `.git` directory out of the artifact. Put the anonymous URL in a
standalone `CODE_LINK.txt` file when the venue requests an external link.

The top-level README should use a neutral project name and include:

1. the software and CUDA/PyTorch/Triton versions;
2. the supported GPU and any required hardware limitation;
3. one command for each reported table or figure;
4. expected output filenames and the acceptance checks; and
5. a short statement of which results are diagnostic or hardware-specific.

Do not include author names, affiliations, e-mail addresses, grants, lab names,
hostnames, usernames, absolute home/scratch paths, scheduler job IDs, or links to
an identifying profile. Check source, README files, notebooks, generated JSON and
CSV metadata, filenames, PDF properties, image EXIF data, and shell history.

## Creating an anonymous link

For a GitHub repository, freeze the exact review snapshot first. Create a branch
or tag such as `submission-v1` and do not change it after the supplementary
deadline. Export the snapshot without `.git` and inspect it locally. Then use the
public instance of [Anonymous GitHub](https://anonymous.4open.science/): paste the
repository URL, add every identifying term that must be replaced, and share the
resulting `https://anonymous.4open.science/r/<id>` URL. Open that URL in a logged-out
browser and inspect every directory, source file, notebook, PDF, image, and link.
Images and PDFs may be served without textual replacement, so they require a
manual check. The mirror is read-only; retain the exact frozen commit hash in the
private records and in the artifact manifest, but do not expose private paths.

An anonymous URL is not a substitute for sanitizing the repository. In
particular, remove identifying text from the current files and do not expose a
public repository URL that redirects to the author's account. Do not put the
original URL, a DOI containing an author-owned record, analytics/tracking links,
or links that reveal who downloaded the data in the paper.

For private data or a large file, an OSF project can be kept private and shared
with a read-only link. In OSF's project sharing dialog, select **Anonymize** when
creating the view-only link. Confirm that contributors, component names, storage
provider names, and file metadata do not identify the authors.

## Venue policy checks (official pages)

- [NeurIPS Code and Data Submission Guidelines](https://nips.cc/public/guides/CodeSubmissionPolicy): code/data for review must be anonymized; small material can be placed in one supplementary ZIP (<100 MB), and large data may use an anonymous URL. De-anonymize URLs only for the camera-ready version.
- [NeurIPS 2026 Evaluations & Datasets FAQ](https://nips.cc/Conferences/2026/EvaluationsDatasetsFAQ): double-blind submissions should use anonymous hosting; code must be documented and executable, and Anonymous GitHub is listed as an option.
- [ICML 2026 Peer Review FAQ](https://icml.cc/Conferences/2026/PeerReviewFAQ): both the URL and its destination must be anonymous. A link containing an identifying GitHub username can result in desk rejection.
- [CVPR 2026 Author Guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines): external links that expand the submission or undermine anonymity are disallowed; when in doubt, keep the submission self-contained. Do not assume an anonymous link is allowed without checking the current year.
- [CVPR Author Suggested Practices](https://cvpr.thecvf.com/Conferences/2025/AuthorSuggestedPractices): the prior guidance permits an anonymous GitHub link on a branch that will not change after the deadline and asks for the link in a standalone text file in the supplementary ZIP. This is venue/year-specific.
- [OSF project sharing](https://help.osf.io/article/353-welcome-to-projects): a private project can create a read-only view-only link with **Anonymize** enabled.

## Final pre-submission audit

Run the following checks on the exported snapshot (replace the patterns with all
names and institutions that must be hidden):

```bash
rg -n -i '(author|affiliation|e-mail|email|@|/home/|/scratch/|/research/|\\Users\\|ssh|slurm|hostname|github.com/)' .
find . -type d -name .git -prune -o -type f -print | sort
```

Review every match rather than blindly replacing symbols such as `@` that may be
part of source code. Check that all commands work from a clean checkout, that no
command requires a private cluster path, and that the result files say exactly
which rows passed, were diagnostic, or were blocked. A passing kernel smoke test
must not be described as an end-to-end or publication-level result unless the
corresponding receipt has all required correctness, timing, provenance, and
exclusivity gates.

After acceptance, replace the anonymous URL with the public repository URL, add
authors and the appropriate license, and publish the exact source snapshot used
for the camera-ready results.

