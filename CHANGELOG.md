# Changelog

## [Unreleased]

### Added

- Save configs in `{runs_dir}/{run_name}`

## [v0.2.2] - 2025-04-14

### Added

- fourrooms `render_args` added
- cli handle `Optional[Literal]` types
- model evaluation support; `oca run --eval`

### Changed

- Custom env `FourRooms-v0` registered

### Fixed

- fourrooms `step` follows gymnasium return signature

## [v0.2.1] - 2025-04-13

### Changed

- Config inheritance from `ConfigProto`
- Agents: `ActorCriticAgent` added
- `oca run` refactored
- Configured loggers for all available agents, at the very least

### Fixed

- All agents at least run now, with some logging

## [v0.2.0] - 2025-04-13

### Added

- Plots: steps vs episodes
- Paths in `constants.py`
- Agents: `SARSAAgent` added
- `oca` cli

## [v0.1.0] - 2025-04-12

### Added

- Initial release.
- Core functionality from <https://github.com/lweitkamp/option-critic-pytorch>
- Setup, config and cli configured

[unreleased]: https://github.com/AshrithSagar/option-critic/compare/v0.2.2...HEAD
[v0.2.2]: https://github.com/AshrithSagar/option-critic/compare/v0.2.1...v0.2.2
[v0.2.1]: https://github.com/AshrithSagar/option-critic/compare/v0.2.0...v0.2.1
[v0.2.0]: https://github.com/AshrithSagar/option-critic/compare/v0.1.0...v0.2.0
[v0.1.0]: https://github.com/AshrithSagar/option-critic/releases/tag/v0.1.0
