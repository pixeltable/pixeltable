# Critic pass

Challenge the proposed findings against the reviewed code and evidence. Correct
the review; leave the product unchanged unless separately authorized.

For every finding, check:

1. **Requirement:** Does it violate a documented/user requirement or observable
   guarantee? Is an implementation preference being presented as a defect?
2. **Attribution:** Was it introduced, newly exposed or relied on by this change?
   Have upstream work, unchanged debt and explicit tradeoffs been distinguished?
3. **Trigger:** Is the stated input/state reachable? Does the full condition include
   all necessary preconditions, such as failure, authorization or generation match?
4. **Trace:** Do callers, persistence and external effects support the causal path?
   Does a guard elsewhere defeat it? Are side effects already in flight being
   conflated with new stale work?
5. **Evidence:** Do the quote, file/line, reproduction output and commit match?
   Does a mocked interleaving demonstrate the mechanism claimed, or merely assume
   it? Is a source trace being called a live run?
6. **Impact/severity:** Is impact concrete and proportionate? Possible billing,
   availability or data loss remains possible unless observed or established.
7. **Correction:** Does the smallest fix preserve ownership, isolation, accepted
   intent, stable identity and compatibility? Check uncertain commits and failures
   of compensation where relevant; a happy-path fix can introduce a worse defect.
8. **Job:** Does the walk use actual product words and one concrete job? Was Done
   observed, expected from source or not reached? Does the product support return
   and the new work success creates? Broader regression checks still apply.
9. **Coverage:** Does an inventory stand in for inspection? Do test totals belong
   to this revision/environment? Are skipped/setup failures, author claims and
   unexamined regions labeled? Are findings duplicated?

Use short working notes when repairs are needed:

| Where | Unsupported or incorrect claim | Check or rewrite needed |
| --- | --- | --- |

Keep the notes internal unless requested. Strengthen evidence when practical;
otherwise narrow the claim or convert it to a question. Do not manufacture a
finding to justify the time spent reviewing. Deliver the draft that survives.
