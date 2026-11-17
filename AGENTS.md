# tabs: agent start instructions

Read `docs/CODEX_HANDOFF.md` before continuing work. For Windows migration, also
read `docs/WINDOWS_HANDOFF.md`. Preserve existing edits and inspect `git status`.

- Work in this checkout on `improve-playable-tabs`; do not reapply earlier fixes.
- Use the project's Python 3.11 virtual environment. Do not copy a Linux `.venv`
  onto Windows. Test with `python -m unittest discover -s tests -v` using that venv.
- The user prefers verified changes committed and pushed to the existing GitHub
  work branch. Check the remote first; never force-push or push to `main`.
- Keep uploaded music/video, stems, models, caches, credentials and private
  handoff archives out of Git. Private inputs belong under ignored `local-input/`.
- Do not publish a cloud environment or deploy the application without a request.
- Compare transcription/separation changes under identical audio, BPM, offset,
  tuning, capo, difficulty and output mode. Report gains AND losses. The existing
  short excerpt was used during development; it is not independent validation.
- Preserve guitar-off backing + synthesized final TAB as the only playback mix.
  Keep TAB, MIDI, PDF and synth based on the same canonical arranged notes.
- Use new dependencies/models in a separate venv first. The external repositories
  listed in `docs/REPOSITORY_CANDIDATES.md` were researched, not integrated.
