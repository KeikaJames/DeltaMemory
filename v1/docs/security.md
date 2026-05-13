# Security Policy and Responsible Use Protocol

Effective Date: 2026-MAY-12

This Security Policy and Responsible Use Protocol (this "Policy") applies to this
repository and to all source code, documentation, examples, scripts, models,
configuration files, tensor banks, injection mechanisms, security utilities,
derivative works, and associated materials made available in or through it
(together, the "Repository").

The Repository includes, without limitation, Mneme, AttenNativeBank / ATB,
DeltaMemory components, tensor-bank tooling, audit utilities, role-checking
scaffolds, encryption helpers, and any related research artefacts.

The Repository is made available solely as a research prototype for legitimate
research, auditing, interpretability, model-safety evaluation, defensive
analysis, and study of large language model behaviour.

Use of the Repository is subject to both:

1. the MIT License included in the Repository; and
2. this Policy.

By accessing, copying, modifying, executing, deploying, distributing, publishing,
or otherwise using any part of the Repository, each user is deemed to have read,
understood, and agreed to comply with this Policy.

---

## 1. Status of the Repository

The Repository is an experimental research prototype.

It is provided for the study of LLM hidden states, attention-layer behaviour,
tensor banks, injection auditing, and related model-safety mechanisms. It has not
been designed, reviewed, certified, or warranted for production, commercial,
regulated, public-facing, or user-impacting deployment.

No representation is made that the Repository is safe, complete, secure,
accurate, suitable for any particular use case, compliant with any particular
legal regime, or fit for deployment in any environment.

Operators and users must conduct their own technical, security, legal, ethical,
and operational assessments before any use.

---

## 2. High-Risk Capability Notice

ATB / AttenNativeBank and associated mechanisms may directly interact with,
alter, steer, or otherwise influence attention layers, hidden states, tensor
banks, steering vectors, projection tensors, or gate tensors of large language
models.

Such functionality is inherently sensitive. Improper, negligent, reckless, or
malicious use may result in serious adverse consequences, including, without
limitation:

- unauthorised or unaudited mutation of model hidden states;
- degradation, circumvention, or manipulation of model safety behaviour;
- generation of unlawful, harmful, deceptive, abusive, unsafe, or unethical
  outputs;
- fraud, scams, phishing, impersonation, social engineering, or other deceptive
  conduct;
- unsafe deployment in production, public-facing, or user-impacting systems;
- breach of applicable laws, regulations, contractual obligations, institutional
  rules, platform rules, or third-party rights.

The publication of code, examples, documentation, test cases, or technical
explanations in the Repository shall not be construed as authorisation,
encouragement, inducement, approval, or endorsement of any misuse.

---

## 3. Permitted Purpose

The Repository may be used only for lawful, ethical, responsible, and authorised
purposes, including:

- research into LLM internals, hidden states, attention behaviour, and model
  steering;
- interpretability, auditing, and safety evaluation;
- defensive testing under proper authorisation;
- reproducible academic or engineering experimentation;
- development of safeguards, monitoring, evaluation, and abuse-prevention
  mechanisms.

Any use outside the foregoing permitted purpose must be assessed by the relevant
operator or user for legality, safety, ethics, and compliance before use.

---

## 4. Prohibited Uses

A user must not use, assist the use of, or permit the use of the Repository, or
any derivative work, for any unlawful, harmful, fraudulent, abusive, deceptive,
reckless, or unethical purpose.

Without limitation, the following uses are prohibited.

### 4.1 Fraud, Scams, and Deception

The Repository must not be used to conduct, facilitate, automate, optimise,
conceal, or scale:

- fraud, scams, phishing, smishing, vishing, social engineering, identity theft,
  impersonation, or credential theft;
- deceptive customer-service, legal, medical, financial, governmental, or
  institutional communications;
- fake notices, fabricated official documents, misleading evidence, forged
  records, or false representations;
- manipulation of users, models, services, platforms, institutions, or systems
  into producing unauthorised, deceptive, or harmful outcomes.

### 4.2 Unlawful Conduct

The Repository must not be used in any manner that violates, or is intended or
likely to facilitate violation of:

- the laws and regulations of Mainland China;
- the laws and regulations of the user's location;
- the laws and regulations of the deployment location;
- the laws and regulations applicable to any affected person, data subject,
  system, platform, institution, service, or jurisdiction.

This includes, without limitation, cybercrime, unauthorised access, data theft,
privacy violations, financial crime, intellectual-property infringement, illegal
surveillance, illegal data processing, illegal export or sanctions evasion, and
unlawful content generation.

### 4.3 Harmful Model Manipulation

The Repository must not be used to:

- bypass, weaken, disable, evade, or degrade safety mechanisms, safeguards,
  access controls, policy filters, or monitoring systems;
- force, induce, optimise, or steer a model to generate unlawful, harmful,
  abusive, deceptive, unsafe, or unethical content;
- develop or operate systems primarily intended to jailbreak, exploit,
  manipulate, subvert, or destabilise AI systems;
- deploy injection mechanisms in a manner that creates uncontrolled, misleading,
  unsafe, or unaudited model behaviour.

### 4.4 Harmful Content Generation

The Repository must not be used to generate, optimise, distribute, or facilitate
content involving:

- violence, threats, harassment, abuse, coercion, or intimidation;
- exploitation or manipulation of vulnerable persons;
- extremist, terrorist, or criminal assistance;
- sexual exploitation, non-consensual sexual content, or content involving
  minors;
- instructions for wrongdoing, evasion, concealment, or unlawful operational
  planning;
- content that breaches applicable legal, ethical, institutional, contractual,
  or platform standards.

### 4.5 Unsafe or Unauthorised Deployment

The Repository must not be deployed in production, commercial, public-facing,
regulated, safety-critical, or user-impacting environments unless the operator
has implemented appropriate safeguards, including, as applicable:

- legal and ethical review;
- security review and threat modelling;
- access control and least-privilege enforcement;
- authentication and authorisation controls;
- audit logging and retention;
- monitoring and abuse detection;
- incident-response processes;
- output review and human oversight;
- key management and secret-handling controls;
- compliance with all applicable laws, rules, and obligations.

---

## 5. User and Operator Responsibilities

Each user and operator is solely responsible for:

- ensuring that their use is lawful, ethical, authorised, and responsible;
- obtaining all necessary rights, permissions, approvals, licences, and consents;
- complying with all applicable laws, regulations, contractual obligations,
  institutional rules, platform rules, and third-party rights;
- assessing the safety, security, and reliability of any system using the
  Repository;
- reviewing model behaviour and outputs before relying on or exposing them;
- preventing misuse by employees, contractors, agents, customers, users,
  downstream recipients, or other third parties;
- implementing adequate access control, rate limiting, monitoring, logging,
  abuse detection, and incident response;
- protecting keys, tensor banks, checkpoints, logs, derived artefacts, and other
  sensitive materials;
- preserving all required license notices, copyright notices, and policy notices
  when copying, modifying, or redistributing the Repository.

---

## 6. Security Scaffold

Mneme v0.4 provides operator-facing primitives for auditing injections,
encrypting small tensor banks at rest, and stubbing role checks around sensitive
bank operations.

This security layer is a scaffold only. It is not a complete security boundary,
production-grade access-control system, identity-management system, compliance
framework, or assurance mechanism.

Operators must supply, test, and maintain all production controls required for
their own environments.

---

## 7. Threat Model

### 7.1 Covered

The Repository provides primitives intended to assist with:

- accidental or unaudited mutation of hidden states by CAA, SCAR, or LOPI;
- plaintext tensor-bank files copied from disk;
- missing role checks before bank load, bank store, injection, or key rotation
  workflows;
- basic auditability of sensitive injection and bank operations.

### 7.2 Excluded

The Repository does not provide protection against:

- compromised Python process memory;
- compromised model weights;
- compromised tokenizer;
- compromised host kernel;
- compromised operating system, container, runtime, or dependency chain;
- malicious code with access to the Fernet key;
- compromised or negligent operators;
- misuse by authorised users;
- identity-provider token validation;
- unsafe downstream deployment;
- unlawful, harmful, or unethical use by users or third parties.

Operators must provide any required external authentication, authorisation,
secrets management, infrastructure hardening, deployment controls, audit review,
and legal-compliance processes.

---

## 8. Audit Log Format

`deltamemory.security.AuditLogger` writes JSON lines to a file path or callback
sink.

Each event has the following structure:

```json
{
  "ts_ns": 0,
  "event_type": "inject|bank_load|bank_store|access_denied",
  "injector": "caa|scar|lopi|null",
  "layer": 0,
  "alpha": 1.0,
  "signal_summary": {
    "steer_norm": 0.0,
    "drift_ratio": 0.0,
    "gate_mean": 1.0
  },
  "vector_hash": "sha256:<64 hex chars>",
  "actor": null,
  "request_id": null
}
```

For injection events, `vector_hash` is computed over the raw bytes of the CAA
steering vector, SCAR projection tensor, or LOPI gate tensor after contiguous CPU
materialisation.

`alpha=0` paths short-circuit before audit emission to preserve the bit-equal
no-op contract.

Audit logs may contain operationally sensitive metadata. Operators should
protect audit logs by appropriate access controls, retention limits, integrity
controls, and confidentiality measures.

---

## 9. Encrypted Bank Storage

Use the optional security extra:

```bash
pip install 'deltamemory[security]'
```

Example:

```python
from cryptography.fernet import Fernet
from deltamemory.security import load_encrypted, save_encrypted

key = Fernet.generate_key()

save_encrypted({"layer0": tensor}, "bank.enc", key)
bank = load_encrypted("bank.enc", key)
```

`torch.load` is called with `weights_only=True`.

Keys are never logged.

Wrong, invalid, missing, or unauthorised keys raise `BankAuthError`.

Encrypted storage reduces the risk of plaintext tensor-bank files being copied
from disk. It does not protect tensor banks after decryption in process memory.

---

## 10. Key Management

Operators should:

- generate and rotate keys outside the Repository;
- store keys in an HSM, cloud KMS, sealed-secret system, or equivalent secrets
  management mechanism;
- pass keys to the runtime through short-lived environment injection or a
  secrets-manager client;
- restrict key access by role, environment, service identity, and operational
  need;
- rotate keys following suspected exposure or unauthorised access;
- audit key access where reasonably practicable;
- avoid committing keys, decrypted bank dumps, encrypted fixtures containing
  real secrets, or any equivalent sensitive material;
- avoid printing, logging, or exposing keys in diagnostics, stack traces, or
  telemetry.

---

## 11. RBAC Integration Guide

`AccessGuard.check(operation, role, actor=...)` implements a local role
hierarchy:

- `bank_load`: `READER`
- `inject`: `READER`
- `bank_store`: `WRITER`
- `rotate_key`: `ADMIN`

This guard does not verify identity-provider tokens.

A production operator should:

1. authenticate each request with an external identity provider, such as OIDC,
   SAML, JWT, or an equivalent mechanism;
2. validate issuer, audience, expiry, signature, revocation status, and other
   relevant token properties where applicable;
3. map IdP claims, groups, service identities, or other trusted attributes to
   `Role.READER`, `Role.WRITER`, or `Role.ADMIN`;
4. call `AccessGuard.check(...)` before load, store, inject, or key rotation;
5. attach an `AuditLogger` so denied operations produce `access_denied` events;
6. review audit logs for suspicious, anomalous, or unauthorised activity.

---

## 12. No Authorisation for Misuse

Nothing in the Repository, the MIT License, this Policy, the documentation, the
examples, or any technical artefact authorises any person to:

- violate any law or regulation;
- infringe the rights of any person;
- bypass security controls;
- bypass model, platform, or service policies;
- access systems, models, services, accounts, data, or networks without
  authorisation;
- generate unlawful, harmful, deceptive, abusive, unsafe, or unethical content;
- deploy unsafe systems affecting real users.

Technical capability must not be treated as permission.

---

## 13. Redistribution and Derivative Works

Any person who copies, modifies, forks, packages, publishes, redistributes, or
builds upon the Repository must preserve, to the extent applicable:

- the original MIT License notice;
- copyright notices;
- this Policy;
- the research-prototype notice;
- the prohibited-use restrictions;
- the no-warranty and no-liability disclaimers.

Downstream recipients should be given reasonable notice that use of the
Repository and derivative works is subject to this Policy.

---

## 14. Vulnerability, Safety, and Misuse Reporting

Security issues, suspected misuse, unsafe behaviour, or vulnerabilities may be
reported through the Repository's issue tracker or any maintainer-designated
security contact.

Reports should include, where possible:

- affected component;
- version or commit hash;
- environment details;
- reproduction steps;
- expected behaviour;
- observed behaviour;
- potential impact;
- suggested mitigation.

Reports should not include secrets, private keys, access tokens, personal data,
confidential third-party data, or sensitive operational details in public
channels.

---

## 15. Disclaimer of Warranty

THE REPOSITORY IS PROVIDED "AS IS" AND "AS AVAILABLE", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, TITLE,
SAFETY, ACCURACY, CORRECTNESS, RELIABILITY, AVAILABILITY, SECURITY, OR
PRODUCTION SUITABILITY.

WITHOUT LIMITING THE FOREGOING, NO WARRANTY IS GIVEN THAT THE REPOSITORY WILL
OPERATE WITHOUT ERROR, INTERRUPTION, VULNERABILITY, HARMFUL OUTPUT, SECURITY
INCIDENT, DATA LOSS, OR LEGAL OR REGULATORY CONSEQUENCE.

---

## 16. Limitation of Liability

TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE AUTHORS, MAINTAINERS,
CONTRIBUTORS, COPYRIGHT HOLDERS, AND AFFILIATED PARTIES SHALL NOT BE LIABLE FOR
ANY CLAIM, DAMAGE, LOSS, LIABILITY, PENALTY, FINE, COST, EXPENSE, OR
CONSEQUENCE ARISING OUT OF OR IN CONNECTION WITH THE REPOSITORY, INCLUDING ANY
USE, MISUSE, MODIFICATION, DEPLOYMENT, REDISTRIBUTION, RELIANCE, OUTPUT,
DERIVATIVE WORK, OR DOWNSTREAM SYSTEM.

THIS LIMITATION APPLIES TO ALL FORMS OF LIABILITY, WHETHER IN CONTRACT, TORT,
NEGLIGENCE, STRICT LIABILITY, STATUTE, EQUITY, OR OTHERWISE, AND WHETHER OR NOT
ANY PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS OR DAMAGE.

WITHOUT LIMITATION, THE AUTHORS, MAINTAINERS, CONTRIBUTORS, COPYRIGHT HOLDERS,
AND AFFILIATED PARTIES SHALL NOT BE LIABLE FOR:

- unlawful, harmful, fraudulent, abusive, deceptive, or unethical use by users
  or third parties;
- harmful, false, misleading, illegal, or unsafe model outputs;
- security incidents, privacy violations, data loss, or unauthorised access;
- financial loss, business interruption, reputational harm, regulatory action,
  enforcement action, or third-party claims;
- unsafe or unauthorised deployment;
- violation of Mainland Chinese law;
- violation of the laws of the user's location;
- violation of the laws of the deployment location;
- violation of the laws of any other applicable jurisdiction.

Each user assumes full responsibility for their own use and for all consequences
arising from that use.

---

## 17. No Insurance, Indemnity, or Support Obligation

The authors, maintainers, contributors, copyright holders, and affiliated parties
provide no insurance, guarantee, indemnity, hold-harmless undertaking,
maintenance obligation, operational assurance, security assurance, service-level
commitment, monitoring obligation, update obligation, or support obligation.

They have no obligation to:

- review user deployments;
- provide security updates;
- monitor misuse;
- provide technical support;
- ensure legal compliance for users;
- compensate users or third parties for any harm, loss, damage, penalty, cost,
  or expense.

---

## 18. Relationship to the MIT License

The Repository includes the MIT License.

The MIT License governs copyright permissions in the covered code.

This Policy governs responsible access, use, deployment, redistribution,
operation, and conduct relating to the Repository and derivative works.

In the event of ambiguity, the MIT License shall continue to apply to copyright
permission as stated in that license, and this Policy shall apply to conduct,
responsible use, security expectations, and risk allocation to the maximum
extent permitted by applicable law.

---

## 19. Severability and Preservation

If any provision of this Policy is held to be invalid, illegal, or unenforceable
by a court, regulator, arbitral tribunal, platform, or other competent authority,
that provision shall be limited or severed to the minimum extent required.

The remaining provisions shall continue in full force and effect to the maximum
extent permitted by applicable law.

No failure or delay by any maintainer in exercising any right, power, or remedy
under this Policy shall operate as a waiver of that right, power, or remedy.
