# Project Branches

## Current Branches

- **main**: Original main branch with the initial project structure
- **experiment**: Development branch with all experimental features implemented so far
- **v0.1.10-stable**: Stable release branch capturing version 0.1.10 with complete working functionality
- **future-improvements**: Branch for ongoing development and new features

## Branch Descriptions

### main
The original main branch containing the initial project structure. This serves as the stable foundation of the project.

### experiment
The development branch where all experimental features have been implemented and tested. This branch contains:
- Real-time ECG simulation
- Synthetic ECG data generation
- ECG classification algorithms
- Real-world ECG data integration
- Continuous playback features

### v0.1.10-stable
This branch represents a stable snapshot of version 0.1.10 of the ECG Viewer project. It includes:
- Complete ECG simulation functionality
- Real-world ECG data import and visualization
- Continuous playback of ECG signals
- Classification and analysis tools
- All features described in the PROJECT_SUMMARY.md file

### future-improvements
This is the active development branch for all future enhancements. Planned improvements include:
- Machine learning integration for improved classification
- Multi-lead ECG support
- Enhanced clinical decision support
- Remote monitoring capabilities
- UI/UX improvements
- Performance optimizations

## Working with Branches

### Viewing Available Branches
```
git branch -a
```

### Switching Between Branches
```
git checkout <branch-name>
```

### Creating a New Feature Branch
```
git checkout -b feature/<feature-name>
```

### Merging Completed Features
```
git checkout future-improvements
git merge feature/<feature-name>
```

## Release Strategy

1. Develop new features in dedicated feature branches
2. Merge completed features into the `future-improvements` branch
3. Test thoroughly in the `future-improvements` branch
4. When a new version is ready for release, create a new stable branch (e.g., `v0.1.11-stable`)
5. Document all changes in `CHANGELOG.md` 