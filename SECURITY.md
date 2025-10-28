# Security Policy

## Reporting Security Vulnerabilities

The GIFT framework is an open-source scientific project. While it's primarily a research framework rather than a production system, we take security seriously to ensure the integrity of scientific computations and to protect users.

### How to Report

If you discover a security vulnerability in GIFT, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Send an email describing the vulnerability to the repository maintainer
3. Or use GitHub's private security advisory feature: https://github.com/gift-framework/GIFT/security/advisories/new

### What to Include

Please include the following information in your report:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)
- Your contact information (if you wish to be credited)

### Response Timeline

- We aim to acknowledge receipt within 48 hours
- We will provide an initial assessment within 7 days
- We will work with you to understand and address the issue
- Once fixed, we will credit you in the release notes (unless you prefer to remain anonymous)

### Scope

Security concerns we care about:
- Code execution vulnerabilities in computational modules
- Data integrity issues that could affect scientific results
- Dependencies with known security vulnerabilities
- Issues that could compromise reproducibility

### Scientific Integrity

If you discover issues that affect the scientific validity of results (but aren't security vulnerabilities), please:
- Open a regular GitHub issue with the "bug" label
- Provide detailed information about the discrepancy
- Include reproduction steps if possible

We treat scientific accuracy with the same seriousness as security.

## Supported Versions

Currently, we support the latest version on the `main` branch. As the project matures, we will establish a formal versioning and support policy.

