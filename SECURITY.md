# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3.0 | :x:                |

## Reporting a Vulnerability

We take the security of scikit-plot seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

1. **Do not** open a public issue on GitHub
2. Email the maintainers at: reiichiro.s.nakano@gmail.com
3. Include the following information:
   - Type of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

### What to Expect

- **Acknowledgment**: You will receive an acknowledgment within 48 hours
- **Assessment**: We will assess the vulnerability within 7 days
- **Fix**: We aim to release a fix within 30 days for confirmed vulnerabilities
- **Disclosure**: We will coordinate disclosure with you

### Security Best Practices

When using scikit-plot:

1. **Input Validation**: Always validate input data before passing to plotting functions
2. **File Handling**: Be cautious when saving plots to files, especially in web applications
3. **Dependencies**: Keep dependencies (matplotlib, scikit-learn, scipy) up to date
4. **Environment**: Use virtual environments to isolate your project

### Known Security Considerations

- scikit-plot processes data through matplotlib and scikit-learn
- No direct network operations are performed
- No execution of arbitrary code from input data
- File operations are limited to plot saving (controlled by matplotlib)

### Dependency Security

We use automated tools to monitor dependencies:

- [Dependabot](https://github.com/dependabot) for automated dependency updates
- [GitHub Security Advisories](https://github.com/advisories) for vulnerability alerts
- Regular security scans with `safety` and `pip-audit`

### Security Updates

Security updates will be released as patch versions and announced via:

- GitHub Security Advisories
- Release notes
- PyPI package updates

---

Thank you for helping keep scikit-plot secure! 🔒
