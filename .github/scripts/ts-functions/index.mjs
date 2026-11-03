// Reads a JSON list of TypeScript and JavaScript file paths on stdin, and prints the cyclomatic complexity and source
// lines of each function in them as JSON: {path: [{name, start, end, complexity, sloc}]}.
//
// The counting is adapted from slopcheck (https://github.com/godagoo/slopcheck-deslop, src/collect.ts),
// Copyright (c) 2026 Goda Go, MIT License; see LICENSE-slopcheck. Changes: anonymous functions count too, named by
// their first line, and methods carry their class name.
import { readFileSync } from 'node:fs';
import { Node, Project, SyntaxKind } from 'ts-morph';

const CALLABLE_KINDS = new Set([
  SyntaxKind.FunctionDeclaration,
  SyntaxKind.FunctionExpression,
  SyntaxKind.ArrowFunction,
  SyntaxKind.MethodDeclaration,
  SyntaxKind.Constructor,
  SyntaxKind.GetAccessor,
  SyntaxKind.SetAccessor,
]);

const SHORT_CIRCUIT = new Set([
  SyntaxKind.AmpersandAmpersandToken,
  SyntaxKind.BarBarToken,
  SyntaxKind.QuestionQuestionToken,
]);

// McCabe complexity: 1, plus 1 per decision point. A nested function keeps its own complexity.
function cyclomatic(fn) {
  let cc = 1;
  fn.forEachDescendant((node, traversal) => {
    if (CALLABLE_KINDS.has(node.getKind())) {
      traversal.skip();
      return;
    }
    switch (node.getKind()) {
      case SyntaxKind.IfStatement:
      case SyntaxKind.ConditionalExpression:
      case SyntaxKind.ForStatement:
      case SyntaxKind.ForInStatement:
      case SyntaxKind.ForOfStatement:
      case SyntaxKind.WhileStatement:
      case SyntaxKind.DoStatement:
      case SyntaxKind.CatchClause:
      case SyntaxKind.CaseClause: // `default:` is a DefaultClause, which decides nothing
        cc += 1;
        break;
      case SyntaxKind.BinaryExpression:
        if (SHORT_CIRCUIT.has(node.getOperatorToken().getKind())) cc += 1;
        break;
    }
  });
  return cc;
}

// Non-blank lines that do not start with a comment.
function sloc(fn) {
  return fn
    .getText()
    .split('\n')
    .filter((line) => {
      const t = line.trim();
      return t.length > 0 && !t.startsWith('//') && !t.startsWith('*') && !t.startsWith('/*');
    }).length;
}

function classPrefix(member) {
  const cls = member.getParent();
  if (!Node.isClassDeclaration(cls) && !Node.isClassExpression(cls)) return '';
  return `${cls.getName() ?? '(anonymous class)'}.`;
}

function nameOf(fn, lines) {
  if (Node.isConstructorDeclaration(fn)) return `${classPrefix(fn)}constructor`;
  const own = fn.getName?.();
  if (own) return classPrefix(fn) + own;
  const parent = fn.getParent();
  if (Node.isVariableDeclaration(parent) || Node.isPropertyAssignment(parent)) return parent.getName();
  if (Node.isPropertyDeclaration(parent)) return classPrefix(parent) + parent.getName();
  if (Node.isExportAssignment(parent)) return 'default';
  // The first line survives edits above the function better than its line number. The comment table cannot show
  // backticks or pipes.
  const first = lines[fn.getStartLineNumber() - 1].trim().slice(0, 60).replaceAll('`', '').replaceAll('|', '/');
  return `(anonymous) at ${first}`;
}

const project = new Project({
  skipAddingFilesFromTsConfig: true,
  skipFileDependencyResolution: true,
  compilerOptions: { allowJs: true },
});
const result = {};
for (const path of JSON.parse(readFileSync(0, 'utf8'))) {
  const sourceFile = project.addSourceFileAtPath(path);
  const lines = sourceFile.getFullText().split('\n');
  const functions = [];
  sourceFile.forEachDescendant((node) => {
    if (!CALLABLE_KINDS.has(node.getKind())) return;
    functions.push({
      name: nameOf(node, lines),
      start: node.getStartLineNumber(),
      end: node.getEndLineNumber(),
      complexity: cyclomatic(node),
      sloc: sloc(node),
    });
  });
  result[path] = functions;
}
process.stdout.write(JSON.stringify(result));
