# Agent Instructions

## Logging

- All Python code **must** use the standard `logging` module for output.
- `print()` is forbidden in production code. Replace any existing `print()` calls with the appropriate log level (`logger.debug`, `logger.info`, `logger.warning`, `logger.error`).
- Each module must obtain its own logger at the top of the file:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  ```

## Language

- All responses, comments, docstrings, and commit messages must be written in **Chinese (Simplified)** or **English**.
- Do not use any other language.

## Code Comments

- Complex logic, non-obvious algorithms, and critical business rules **must** have inline comments or docstrings explaining *why*, not just *what*.
- Simple, self-explanatory code does not require comments.

## Multi-File Changes

- If a task requires modifying or creating **more than 2 files**, first output a written plan that lists:
  1. Each file to be changed or created.
  2. A brief description of what will be changed and why.
- Wait for explicit user confirmation before implementing any of the changes.
