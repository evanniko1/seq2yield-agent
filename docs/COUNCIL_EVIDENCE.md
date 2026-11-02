# COUNCIL_EVIDENCE.md — does a role-based council earn its place?

A critical review of the 2025–26 evidence on multi-agent LLM deliberation, applied to this
project's own council, plus the architectural consequence (the proposer seam, DECISIONS #79).

Written 2026-11-02. Every number below is attributed; where a figure comes from a secondary source
it says so, because much of this literature is preprints and blog summaries and the primary papers
do not always say what the summaries claim.

---

## 1. What this project's council actually is

| | count |
|---|---|
| roles defined in `configs/agent_roles.yaml` | **13** |
| `enabled: true` | **8** |
| of those, in the **council** (deliberating) side | **5** — `principal_investigator`, `proposal_generator`, `biology_architect`, `adversarial_critic`, `chair` |
| on the **execution** side (post-decision) | **3** — `ml_engineer`, `patch_reviewer`, `postmortem_synthesizer` |
| live **critics** | **1** (`adversarial_critic`) |
| retired but kept for ablation (`enabled: false`) | **5** — `modeling_reviewer`, `methodology_reviewer`, `biology_reviewer`, `transformer_reviewer`, `doe_strategist` |

So the honest description is **not** "an 11-role council". It is a five-stage pipeline with a single
adversarial critic, plus three post-decision roles — the result of the collapse recorded in the
direction debate. There is also one role invoked in code that is **not declared**:
`methodology_critic` (`scripts/run_agent_loop.py:283`, `src/agents/methodology_critic.py`) passes
`role="methodology_critic"` into model calls, and `roles.persona()` returns `""` for it because
`agent_roles.yaml` has no such entry.

**Deliberation depth.** `debate_rounds` defaults to 1 and reviewers score independently, so this is
a *pipeline with a critique stage*, not a debate. That matters for which literature applies: almost
all the multi-agent-debate results below are about iterated cross-talk, which this project does not
do.

### 1a. A governance gap found while counting

`configs/provider_policy.yaml` `role_policy` has four entries — `chair`, `methodology_reviewer`,
`patch_reviewer`, `runspec_compiler` — and **no `default`**. Consequences, all verified against
`router._build`:

- `require_direct_provider` is enforced for exactly **2 of the 8 enabled roles** (`chair`,
  `patch_reviewer`). Five roles declared `provider_class: authority` in `agent_roles.yaml` —
  `adversarial_critic`, `biology_architect`, `ml_engineer`, `principal_investigator` — have no
  policy entry, so the direct-provider filter never applies to them.
- AGENTS.md §5 names "methodology/statistical reviewer" as an authority role requiring a direct
  provider. That role is now **disabled**: the collapse folded it into `adversarial_critic` and the
  policy entry stayed with the retired role. **The council's only live critic therefore does not
  carry the guarantee the document promises.**
- `runspec_compiler` has a policy entry but is not a declared role — dead config. (AGENTS.md treats
  "Chair / RunSpec Compiler" as one role.)
- `methodology_critic` appears in neither file.

This is a documentation-vs-code divergence of the kind ST-12 covers, but with teeth: the
`require_direct_provider` rule exists for schema reliability and auditability, and the role it most
matters for is the one adjudicating methodology. Filed as **AG-24**.

---

## 2. The case against councils, and how much of it bites here

### 2.1 At equal compute, more agents is not obviously better
The central critique is budget confounding. [Single-Agent LLMs Outperform Multi-Agent Systems on
Multi-Hop Reasoning Under Equal Thinking Token Budgets](https://www.alphaxiv.org/abs/2604.02460)
reports that once reasoning-token budgets are matched, single-agent setups match or beat
multi-agent ones, and debate configurations rarely exceeded a single agent using the same total
tokens. The original debate result (Du et al., ICML 2024) never presented an equal-budget baseline.
A widely-cited comparison, "Stop Overvaluing Multi-Agent Debate", is summarised as finding plain
chain-of-thought beating debate in >80% of configurations across five debate methods and nine
benchmarks — **I have not verified that against the primary paper** and quote it only as the
direction the secondary literature points.

*Bite here:* **high, but indirect.** This project's council does not debate, so the specific MAD
results do not transfer. What does transfer is the accounting discipline: nobody has measured
whether this council beats a cheaper proposer on this project's own metric. That is now
measurable — see §4.

### 2.2 Debate can actively hurt, especially with heterogeneous strength
[Talk Isn't Always Cheap](https://arxiv.org/html/2509.05396v1) finds multi-agent debate sometimes
degrades performance below a single agent, worst when a weaker agent joins a stronger one.
[A layered analysis](https://arxiv.org/html/2609.08016v2) finds no measurable final-answer quality
gain on open-ended questions under its evaluator. Plain majority voting beating decentralised
debate (0.99 vs 0.76 on arithmetic with Qwen) is reported in a secondary summary of Choi, Zhu & Li.

*Bite here:* **direct and specific.** This project's provider policy deliberately mixes tiers —
`diversity` (Ollama, OpenRouter) for brainstorming and `authority` (Anthropic, OpenAI) for
adjudication. The weaker-agent-harms-stronger finding is precisely the configuration risk. It is
mitigated by structure (the critic and chair are authority-tier and the generator does not get a
vote on admissibility) — but §1a shows the critic's authority tier is **not actually enforced**, so
the mitigation is partly nominal.

### 2.3 Role personas do not reliably buy accuracy
Wharton's [Prompting Science Report 4](https://gail.wharton.upenn.edu/research-and-insights/playing-pretend-expert-personas/)
(Dec 2025) tested expert personas on GPQA Diamond and MMLU-Pro across six models: personas do not
reliably improve accuracy, and low-knowledge personas often reduce it. [Earlier work on 162
personas](https://arxiv.org/html/2311.10054v3) found adding a persona does not necessarily improve
performance on objective tasks.

*Important limit, and the reason this is not decisive:* these test **single-prompt persona framing**
— "you are an expert biologist" prepended to one call. They do not test **functional
decomposition**, where each agent has a different input, a different output schema and a different
authority. "Reviewer personas don't help" and "separating propose-from-adjudicate doesn't help" are
different claims, and only the first is supported. This project's roles are mostly functional, not
personal: the chair's distinguishing feature is that it is the only role that may select, not that
it is told it is a chair.

*Bite here:* **moderate.** It argues against the retired five-specialist panel (five personas, one
input, overlapping rubrics) and barely touches the generator/critic/chair split. The collapse
already acted on this.

### 2.4 Multi-agent systems fail in characteristic, structural ways
[MAST](https://arxiv.org/pdf/2503.13657) (Cemri, Pan, Yang et al., NeurIPS 2025) derives 14 failure
modes in three categories — system design, inter-agent misalignment, **task verification** — from
grounded-theory analysis of 150+ traces, later extended past 1,600. Relevant modes: FM-1.2 *Disobey
Role Specification*; 3.2 *No or Incomplete Verification*; 3.3 *Incorrect Verification*. The paper
reports 3.2 and 3.3 appearing **even in successful runs**. A follow-on coding-agent study reports
MAST's authors finding prompt changes gave limited gains on persistent failure classes while
**redesigned verification topology and modular roles helped more**.

*Bite here:* **this is the most useful result for this project, and it is supportive.** The whole
architecture — a deterministic harness that agents cannot override, schema-validated outputs,
protected files, an audit log, an approval ledger — is a direct answer to category 3. It also names
the project's live defects: `IM-10` (a failed reviewer replaced by fabricated 3/3/3/3 scores
averaged in as real) is FM-1.2 and FM-3.3 together; `AG-01` (infrastructure failures recorded as
scientific negatives) is 3.2.

### 2.5 Self-verification is weak; external verification is what works
[On the Self-Verification Limitations of LLMs](https://proceedings.iclr.cc/paper_files/paper/2025/file/f3c5e56274140e0420baa3916c529210-Paper-Conference.pdf)
(ICLR 2025) finds performance *collapses* under self-critique and improves substantially with a
sound external verifier — and that simply sampling many guesses beat the self-critique loop.
[Mind the Gap](https://proceedings.iclr.cc/paper_files/paper/2025/file/63943ee9fe347f3d95892cf87d9a42e6-Paper-Conference.pdf)
(ICLR 2025) finds the generation–verification gap grows with verifier capability and shrinks with
generator capability. A survey cites ~64.5% of self-generated errors surviving self-checking across
14 open models while the same errors are caught when presented externally (secondary; unverified).
[Weaver](https://scalingintelligence.stanford.edu/pubs/weaver.pdf) shrinks the gap by combining
multiple *weak* verifiers.

*Bite here:* **this is the strongest argument for this project's shape and against spending more on
the council.** The harness is a *sound external verifier* in the precise sense these papers mean:
it runs code, computes the metric, applies a pre-registered margin rule and can reject the agent
that proposed the work. Every marginal dollar on deliberation buys a better *generator*; the gap
literature says the binding constraint is the *verifier*, and the verifier here is deterministic
and already strong. Weaver also suggests the retired panel has a legitimate afterlife — as an
ensemble of weak verifiers over a fixed artifact, not as five personas arguing.

### 2.6 "AI scientist" systems are bottlenecked on verification, not ideation
[Autonomous Research Agents: A Survey of AI Scientists and the Verification Gap](https://arxiv.org/pdf/2608.05179)
argues current agents are capable co-scientists but not built for autonomous discovery, citing
biased problem selection, missing tacit knowledge, diversity collapse from preference optimisation,
and benchmarks measuring single-turn accuracy rather than closed-loop validity — and notes agent
evaluations routinely omit harness, cost and reproducibility details.
[MLReplicate](https://arxiv.org/html/2605.16616v1) scores six systems on ICML 2025 outstanding
papers: AI-Scientist-v2 best at mean 0.42 / peak 0.71, and concludes the gap to genuine rigour is
substantial. [RECLAIM](https://arxiv.org/html/2609.28850v1) grades *execution evidence* rather than
the agent's self-report and finds the worst case where no code is released: 13 of 132 cells
reproduce.

*Bite here:* **validating, and a warning.** "Grade the execution evidence, not the agent's report"
is this project's run-card discipline. The survey's charge that evaluations omit harness, cost and
reproducibility detail is exactly what this project's run-cards, `ModelCallRecord` logging and
provenance fingerprints exist to prevent. The warning is diversity collapse and biased problem
selection — which maps onto the insight layer's job of *generating* questions rather than selecting
grid cells.

---

## 3. The case **for** keeping a council, stated fairly

Steel-manning, because §2 is one-sided if left alone:

1. **The original debate gains were never overturned**, only re-contextualised. Du et al.'s
   arithmetic result (81.8% vs 67.0%) stands as reported; the critique is that it lacked an
   equal-budget control. At least one source notes debate helps *more* as problems get harder and
   models get weaker.
2. **Functional decomposition ≠ persona prompting.** The negative persona results do not test a
   generator that may not select, a critic that may not patch, and a chair that may not execute.
   That separation is a *governance* property, not a performance one, and this project needs it for
   auditability regardless of whether it raises accuracy: AGENTS.md §2's role boundaries are what
   make the audit log interpretable.
3. **Weak-verifier ensembles do help** (Weaver), which is a real argument for a panel — just
   pointed at verification rather than at ideation.
4. **Heterogeneity is the ingredient that survives.** Where multi-agent gains replicate, the
   literature attributes them to *diversity of generation* rather than role count. The
   `diversity`-tier generator plus `authority`-tier adjudicator split is the cheap version of that.
5. **Coverage and novelty need state, not intelligence.** The PI/question-space machinery exists to
   stop re-proposing settled cells. That is bookkeeping a single agent would also need.

---

## 4. Verdict, and what makes it falsifiable

**The council is plausibly over-built for what it demonstrably adds, and the honest position is
that nobody has measured it.** The five-reviewer collapse was already the right direction. The
remaining five council-side roles are defensible on *governance* grounds (separation of propose /
critique / select is what makes the audit trail meaningful) and **unmeasured** on *performance*
grounds.

The literature's most robust and least contested finding — the verifier is the binding constraint,
not the generator — says the harness is the durable asset. So the architecture should never
hard-wire one deliberation scheme. It now does not: see `src/orchestration/proposer.py`.

**The falsifiable question, now runnable:** does the council beat a deterministic sweep, on this
project's own primary metric, under equal-or-better accounting? `SweepProposer` emits a valid
RunSpec with **zero** model calls and is scored by the same harness, the same two-component
estimator and the same ST-3 margin rule. `ProposalSet.model_calls` makes the budget axis explicit,
which is the exact control the MAD critiques found missing. A null result there is a publishable
finding under AGENTS.md §7, and it is the single most informative experiment this project could run
about its own architecture.

Note what the comparison can and cannot settle: it measures *proposal quality under a fixed
intervention menu*. It cannot measure what the council is actually for in the long run — generating
questions that are not already grid cells. That capability lives in the insight layer, and the
survey literature in §2.6 is unanimous that it, not deliberation, is where discovery comes from.

---

## 5. Sources

- [Single-Agent LLMs Outperform Multi-Agent Systems Under Equal Thinking Token Budgets](https://www.alphaxiv.org/abs/2604.02460)
- [Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate](https://arxiv.org/html/2509.05396v1)
- [What Does Multi-Agent LLM Debate Actually Change? A Layered Analysis](https://arxiv.org/html/2609.08016v2)
- [Why Do Multi-Agent LLM Systems Fail? (MAST)](https://arxiv.org/pdf/2503.13657) · [NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/file/b1041e52d3be19f0a9bc491657488e4a-Paper-Datasets_and_Benchmarks_Track.pdf)
- [On the Self-Verification Limitations of Large Language Models (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/f3c5e56274140e0420baa3916c529210-Paper-Conference.pdf)
- [Mind the Gap: Examining the Self-Improvement Capabilities of LLMs (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/63943ee9fe347f3d95892cf87d9a42e6-Paper-Conference.pdf)
- [Shrinking the Generation-Verification Gap with Weak Verifiers (Weaver)](https://scalingintelligence.stanford.edu/pubs/weaver.pdf)
- [Trust but Verify! A Survey on Verification Design for Test-time Scaling](https://arxiv.org/pdf/2508.16665)
- [Prompting Science Report 4: Expert Personas Don't Improve Factual Accuracy (Wharton)](https://gail.wharton.upenn.edu/research-and-insights/playing-pretend-expert-personas/) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5879722)
- [When "A Helpful Assistant" Is Not Really Helpful: Personas in System Prompts](https://arxiv.org/html/2311.10054v3)
- [Autonomous Research Agents: A Survey of AI Scientists and the Verification Gap](https://arxiv.org/pdf/2608.05179)
- [MLReplicate: Benchmarking Autonomous Research Systems for ML Reproducibility](https://arxiv.org/html/2605.16616v1)
- [RECLAIM: Can Agents Reproduce the Claims of Machine Learning Papers?](https://arxiv.org/html/2609.28850v1)
- [A Comparative Study of MCP and A2A for Inter-Agent Coordination](https://arxiv.org/pdf/2607.23884)
- [A2ABreak: Systematic Security Analysis of the A2A Protocol](https://arxiv.org/pdf/2609.10871)
- [From Multi-Agent Systems and the Semantic Web to Agentic AI: the Web of Agents](https://arxiv.org/pdf/2507.10644)
