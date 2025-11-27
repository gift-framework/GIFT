# Contributing to GIFT Framework

Thank you for your interest in contributing to the Geometric Information Field Theory framework. This document provides guidelines for contributions that maintain scientific rigor while fostering collaborative development.

## Code of Conduct

### Scientific Standards

The GIFT framework maintains strict standards:

1. **Rigor**: All mathematical claims require proof or clear status classification
2. **Reproducibility**: Calculations must be verifiable through provided code
3. **Transparency**: Methods, assumptions, and limitations clearly stated
4. **Honesty**: Report both successes and failures, agreements and tensions
5. **Humility**: Maintain speculative tone appropriate to theoretical work

### Community Principles

- Respectful scientific discourse
- Constructive criticism focused on ideas, not individuals
- Recognition of contributions from all participants
- Open sharing of methods and data
- Acknowledgment of limitations and uncertainties

## Types of Contributions

### 1. Scientific Contributions

**New Predictions**
- Derive additional observables from geometric structure
- Extend framework to new physics sectors
- Propose experimental tests

**Mathematical Improvements**
- Provide rigorous proofs for THEORETICAL status results
- Refine derivations for higher precision
- Identify new exact relations

**Experimental Comparisons**
- Update predictions with latest experimental data
- Perform statistical analysis of agreement
- Identify tensions requiring investigation

### 2. Documentation Contributions

**Clarifications**
- Improve explanations of complex concepts
- Add examples and worked calculations
- Enhance cross-references

**Educational Content**
- Tutorial materials for specific topics
- Visualizations of geometric structures
- Pedagogical presentations

**Translation**
- Language translations of documentation
- Conversion to different formats (LaTeX, HTML)

### 3. Computational Contributions

**Code Improvements**
- Enhance numerical precision
- Optimize performance
- Add new visualization tools

**Verification**
- Independent calculation verification
- Alternative computational approaches
- Error checking and validation

## Contribution Process

### Before Starting

1. **Review Existing Work**
   - Read relevant sections of main paper and supplements
   - Check `docs/FAQ.md` for common questions
   - Search existing issues: https://github.com/gift-framework/GIFT/issues

2. **Open an Issue**
   - Describe proposed contribution
   - Explain motivation and approach
   - Allow community feedback before substantial work

3. **Discuss Approach**
   - Engage with maintainers and community
   - Clarify scope and methodology
   - Align with framework principles

### Making Changes

1. **Fork Repository**
   ```bash
   git clone https://github.com/gift-framework/GIFT.git
   cd gift
   git checkout -b feature/your-contribution
   ```

2. **Follow Structure**
   - Place content in appropriate location (see `STRUCTURE.md`)
   - Use consistent notation (see `docs/GLOSSARY.md`)
   - Apply status classifications (PROVEN, TOPOLOGICAL, etc.)

3. **Document Thoroughly**
   - Explain all steps in derivations
   - Provide references for external results
   - Include computational verification where applicable

4. **Update Related Files**
   - Add to `CHANGELOG.md` under "Unreleased"
   - Update relevant sections in main paper
   - Modify `docs/FAQ.md` if introducing new concepts

### Submitting Contributions

1. **Create Pull Request**
   - Clear title describing contribution
   - Detailed description of changes
   - Reference related issues

2. **Review Process**
   - Maintainers will review for scientific accuracy
   - Community members may provide feedback
   - Iterations may be needed for clarity/rigor

3. **Acceptance Criteria**
   - Mathematical correctness verified
   - Computational results reproducible
   - Documentation complete and clear
   - Consistent with framework principles

## Specific Guidelines

### Mathematical Content

**Notation**
- Follow conventions in `publications/gift_2_2_main.md` Section 1.4
- Define all new symbols on first use
- Use standard notation where possible (see `docs/GLOSSARY.md`)

**Status Classification**
Use appropriate status for all results:
- **PROVEN**: Only for rigorously proven exact relations
- **TOPOLOGICAL**: Direct consequence of manifold structure
- **DERIVED**: Calculated from proven/topological results
- **THEORETICAL**: Theoretical justification, proof incomplete
- **PHENOMENOLOGICAL**: Empirical fit, theory in progress
- **EXPLORATORY**: Preliminary investigation

**Precision**
- Report numerical results with appropriate significant figures
- Include error estimates where applicable
- Compare with experimental values and uncertainties
- State assumptions clearly

### Computational Code

**Style**
- Python code follows PEP 8 guidelines
- Notebook cells have clear markdown explanations
- Functions include docstrings

**Dependencies**
- Use only packages in `requirements.txt`
- Request addition of new dependencies with justification

**Verification**
- Include test cases for new functions
- Verify numerical results against analytical expressions
- Document numerical precision achieved

### Documentation

**Tone**
- Sober and humble, avoiding hype
- Speculative where appropriate
- Balanced presentation of strengths and limitations

**Formatting**
- Markdown for all documentation
- No emojis in technical documents
- Mathematical notation: `\( ... \)` inline, `\[ ... \]` display

**Citations**
- Use consistent citation format
- Include all relevant references
- Prefer peer-reviewed sources

## Review Criteria

### Scientific Review

Contributions are evaluated on:

1. **Correctness**: Mathematical derivations are sound
2. **Reproducibility**: Results can be independently verified
3. **Clarity**: Presentation is comprehensible
4. **Relevance**: Advances the framework meaningfully
5. **Rigor**: Appropriate level of proof/justification

### Technical Review

Code contributions are evaluated on:

1. **Functionality**: Code runs without errors
2. **Efficiency**: Reasonably optimized
3. **Readability**: Well-commented and structured
4. **Testing**: Includes verification of results

## Recognition

Contributors will be:
- Acknowledged in relevant document sections
- Listed in repository contributors
- Cited in publications arising from contributions

Substantial contributions may warrant co-authorship on specific publications, determined case-by-case following standard academic practices.

## Getting Help

### Resources

- **Documentation**: Start with `README.md` and `STRUCTURE.md`
- **Questions**: Open an issue tagged "question"
- **Discussions**: Use GitHub discussions for broader topics

### Contact

- **Repository**: https://github.com/gift-framework/GIFT
- **Issues**: https://github.com/gift-framework/GIFT/issues
- **Pull Requests**: https://github.com/gift-framework/GIFT/pulls

## Scientific Integrity

### Conflicts of Interest

Disclose any potential conflicts:
- Financial interests in related technologies
- Competing theoretical frameworks
- Professional relationships affecting objectivity

### Data and Methods

- Share all data used in analyses
- Provide complete computational methods
- Make code available for verification
- Report negative results alongside positive ones

### Authorship

Follow standard academic practices:
- Substantial intellectual contribution required for authorship
- All authors approve final version
- Authors take responsibility for their contributions

## Examples of Contributions

### Good Contribution Examples

**Example 1: New Prediction**
```
Title: Derive axion mass from K₇ topology

Description: Using the harmonic 3-form structure from Supplement S2,
I derive a prediction for axion mass: m_a = 10^-5 eV * (geometric factor).

Status: THEORETICAL (proof outline provided, full rigor in progress)

Verification: Numerical calculation agrees with formula to 10 digits.
Experimental comparison: Consistent with current bounds m_a < 10^-4 eV.
```

**Example 2: Improved Precision**
```
Title: Refine sin²θ_W calculation using two-loop corrections

Description: Include two-loop corrections in gauge coupling running,
improving sin²θ_W prediction from 0.009% to 0.003% deviation.

Impact: Strengthens agreement with experimental value.
Method: Updated equations in S4_complete_derivations.md with full derivation.
```

### What to Avoid

**Avoid: Unsupported Claims**
```
❌ "This proves string theory is correct"
✓ "This provides evidence consistent with string-theoretic frameworks"
```

**Avoid: Hiding Limitations**
```
❌ Reporting only successful predictions
✓ Documenting both agreements and tensions with experiment
```

**Avoid: Insufficient Documentation**
```
❌ "The result is obviously true"
✓ Providing step-by-step derivation or numerical verification
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License, consistent with the repository license. See `LICENSE` file for details.

## Updates to Guidelines

These guidelines may evolve. Check this document periodically for updates. Major changes will be announced through:
- GitHub repository announcements
- Updates to `CHANGELOG.md`
- Discussion threads for community input

## Questions About Contributing

If anything in these guidelines is unclear:
1. Check `docs/FAQ.md`
2. Open an issue tagged "contributing-question"
3. Refer to `STRUCTURE.md` for organizational details

---

Thank you for helping advance our understanding of the geometric foundations of physics. Your contributions, whether mathematical derivations, computational tools, or documentation improvements, strengthen the scientific community's ability to test and refine fundamental theories.

