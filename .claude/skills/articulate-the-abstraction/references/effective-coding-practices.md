<!-- Marcel's guidelines, reproduced verbatim; typography converted to ASCII. The skill's principles build on them. -->

# Effective Coding Practices

---

From an accounting perspective, code is both an asset and a liability: we expect it to generate value, but each line of code typically also requires additional engineering time after it is initially written, be it for bug fixes or because someone needs to figure out how to take advantage of some library functions you wrote. The collective effort spent on reading a line of code easily exceeds that spent on writing it by an order of magnitude.

Any algorithm can be expressed with maximum clarity, and the goal of your implementation is to make the code as "consumable" by your fellow engineers as possible. The purpose of the following coding guidelines is to highlight practices that minimize the "liability" component of your code and maximize readability.

# Abstraction

Abstractions are the foundation of any codebase; without abstractions, you need to know everything in order to do anything - clearly not a recipe for stellar productivity.

- Every class/struct/type, etc., that you create should reflect an abstraction.
- A good abstraction is self-contained and has a limited, coherent set of responsibilities.
- Moreover, you should be able to articulate that abstraction in a few crisp sentences. If you can't, chances are that you didn't really identify an abstraction, at least not a good one. Good news: you identified a piece of code that can benefit from major improvements! Start over!
- Once an abstraction starts to break down, the code tends to deteriorate progressively, to the point where it is labelled "tech debt".
- For that reason, it's best to evolve abstractions rather than to try to shoehorn new and conflicting requirements into an existing abstraction. If a class can be naturally augmented to capture new requirements, that's the right thing to do (they weren't conflicting in that case). If that cannot be done, it's best to create a new class or set of classes that captures the complete set of requirements.

# Commenting

Comments play a critical role in making code more consumable. To start with, they are the only way to express intent to the reader: there is no way to express 'why' in our languages except through the use of comments.

Comments also act as a check, both for the reader and the writer. If you find it hard to write a clear comment, perhaps your abstraction is not clear? Does the comment you want to write obviously contradict what you just implemented? As you are reviewing the code, are the comments accurate? If not, that's a great sign that the implementation might not be well understood.

- Brevity is a virtue: reading comments takes time, no need to point out the obvious.
- Always write a class comment that contains all relevant info for the user of that class: the abstraction, any non-obvious externally observable behaviour, common usage patterns (if those aren't completely obvious from the interface itself), as well as anything that's out of the ordinary (lack of thread safety, restrictions on usage etc.).
- The same goes for functions: each non-trivial function requires a comment in the header file containing all the relevant info for the caller of that function: its externally observable behavior, including the roles of the parameters and all side effects. (Getters and setters are examples of trivial functions and excluded from that requirement.)
- Always write comments when logic is not self evident either in its intent ("`this loop computes the FFT of the signal in samples_`") or its implementation ("`Find the set of equivalent predicates by computing the connected components of the transfer graph`").
- Always write a comment when you think things are non-obvious ("`We have to copy fragment_params_ here because once we release the lock, the caller may free the original itself`").
- Explicitly point out invariants in your code, especially between member variables. This increases the likelihood that future changes to the code will keep those invariants intact. This should always be done with explanatory comments directly next to the member variables. (In addition, you should use assertions (via `DCHECKs`) to ensure that these invariants hold.)

# Naming

Well-chosen names are very helpful for making a piece of code more accessible. This applies to anything that can be named: functions, variables, classes, structs, typedefs, etc.

- Always use clear, unambiguous names. The more specific, the better.
- Good variable names often reduce the need for explanatory comments.

    Example:

    instead of int64 bytes_; // bytes written

    this will remind the reader what those bytes are: `int64 bytes_written_;`

- Do not embed the type in the variable name ("`my_int_var_`"). Iterators are an exception to this rule ("`hosts_iterator_`"). Another exception is to distinguish otherwise identical variables ("`TNetworkAddress host_addr; string host_addr_str;`").
- It is often tempting to take shortcuts and choose names that are only kind of accurate, because, hey, I know what it means anyway. Resist that temptation and instead invest some time in coming up with the exact right name. If you spend 1 minute but save every person from then on 30 seconds, your team will be ahead.

# Code "Smells"

This is a non-exhaustive list of pathological patterns that should be avoided.

1. **Superfluous Comments** There's a fine line between comments that illuminate and comments that obscure. Are the comments necessary? Do they explain "why" and not "what"? Can you refactor the code or pick better variable names so the comments aren't required? And remember, you're writing comments for people, not machines.
2. **Long Function** All other things being equal, a shorter method is easier to read, easier to understand, and easier to troubleshoot. Refactor long methods into smaller methods if you can, unless these small methods would violate other code smells (e.g., 3, 7, 10).
3. **Long Parameter List** The more parameters a method has, the more complex it is. Limit the number of parameters you need in a given method, or use an object to combine the parameters.
4. **Duplicated Code** Duplicated code is the bane of software development. Stamp out duplication whenever possible. You should always be on the lookout for more subtle cases of near-duplication, too. Don't Repeat Yourself!
5. **Large Class** Large classes, like long methods, are difficult to read, understand, and troubleshoot. Does the class contain too many responsibilities? Can the large class be restructured or broken into smaller classes?
6. **Type Embedded in Name** Avoid placing types in method or variable names; it's not only redundant, but it forces you to change the name if the type changes.
7. **Uncommunicative Name** Does the name of the method succinctly describe what that method does? Could you read the method's name to another developer and have them explain to you what it does? If not, rename it or rewrite it.
8. **Inconsistent Name** Pick a set of standard terminology and stick to it throughout your methods. For example, if you have Open(), you should probably have Close().
9. **Inconsistent Structure** Similar code should look similar. For example, two loops that have the same terminating condition should be structured similarly, conditionals that are equivalent or similar should be structured similarly. Generally, local patterns should be followed to make it easier to see similarities in logic.
10. **Dead Code** Ruthlessly delete code that isn't being used. That's why we have source control systems!
11. **Speculative Generality** Write code to solve today's problems, and worry about tomorrow's problems when they actually materialize. Everyone loses in the "what if.." school of design. You (Probably) Aren't Gonna Need It.
12. **Gratuitous Templatization** Templates should be seen as a measure of last resort, when it is clear that the use case at hand will see measurably better performance, or you really need to stamp out the same code for a number of types. In particular, don't templatize if there is only a single type instantiation needed.
13. **Missing Abstraction** Don't use a gaggle of primitive data type variables as a poor man's substitute for a class. If your data type is sufficiently complex, write a class to represent it. Also, if you always see the same data hanging around together, maybe it belongs together. Consider rolling the related data up into a larger class.
14. **Questionable Inheritance** If you inherit from a class, but never use any of the inherited functionality, should you really be using inheritance?
15. **Entangled Classes** Watch out for classes that spend too much time together, or classes that interface in inappropriate ways, such as accessing each other's internal state. Classes should know as little as possible about each other.
16. **Exposed Internals** Beware of classes that unnecessarily expose their internals. Aggressively refactor classes to minimize their public surface. You should have a compelling reason for every item you make public. If you don't, hide it.
17. **Superfluous Classes** Classes should pull their weight. Every additional class increases the complexity of a project. If you have a class that isn't doing enough to pay for itself, can it be collapsed or combined with another class? Also, if a class is delegating all its work, why does it exist? Cut out the middleman. Beware classes that are merely wrappers over other classes or existing functionality in the framework.
18. **Performance Cliffs** Avoid performance optimization that only work in certain, very specific scenarios and create dramatic performance swings in response to minor variations in input. An example from the DBMS world would be: improving the performance of a specific Where clause predicate, but only implementing that for a subset of the supported data types.

- Psychological safety
    - open criticism is not good
- A debate is not collaborative. Get rid off a debate culture
