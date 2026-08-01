---
name: Bug report
about: Something behaves differently from what the docs or the paper describe
title: ''
labels: bug
assignees: ''
---

**Before filing:** please check [`KNOWN_ISSUES.md`](../../KNOWN_ISSUES.md) —
several surprising behaviours are tracked there deliberately, with the
verification steps already run.

### What happened

<!-- What you observed, and what you expected instead. -->

### How to reproduce

<!-- The exact command or a minimal snippet. -->

```
```

### Which path

- [ ] No-SDV verification path (`python -m logiscag.reproduce --verify`)
- [ ] SDV real path (`python -m logiscag.reproduce --real ...`)
- [ ] Library use (`import logiscag`)

### Environment

```
python --version
pip show logiscag pandas numpy sdv
```

### Traceback and warnings

<!-- Full traceback. Please include any UserWarning text — warnings in this
package are deliberate signals, not noise. -->

```
```

### Does this concern a code↔paper divergence?

<!-- If the code does something the paper doesn't describe (or vice versa),
say so — those are treated as higher priority than ordinary bugs. -->
