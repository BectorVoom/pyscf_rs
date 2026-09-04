//! Parser for the Python-literal basis files under `pyscf/gto/basis/`.
//!
//! A handful of upstream basis sets are stored as a Python MODULE holding one
//! nested-list literal per element rather than as NWChem or Gaussian-94 text —
//! `minao.py`, `iglo3.py`, `faegre_dz.py`. `pyscf/gto/basis/__init__.py:665-676`
//! imports them with `importlib` and reads the module attribute named after the
//! element symbol.
//!
//! The literal shape is exactly the internal basis format:
//!
//! ```text
//! Si = [[0,
//!        [254900, 6.25e-05, -1.66e-05, 4.26e-06],
//!        ...
//!        ],
//!       [1,
//!        [481.5, 1.92e-03, -4.05e-04],
//!        ...
//!        ]]
//! ```
//!
//! i.e. `[[l, [exp, c_0, c_1, …], …], …]` — one outer entry per shell, with the
//! first element of each inner row the exponent and the rest one coefficient per
//! contraction.
//!
//! This parser reads the literal directly. It does NOT execute Python: it
//! strips `#` comments, finds the `SYMBOL =` assignment, and walks the balanced
//! brackets that follow, which is all the format needs.

use pyscf_core::{BasisLoadError, ParsedBasis, ShellSpec};

/// Parse the `symbol` entry out of a Python-literal basis module.
///
/// `symbol` is matched case-insensitively against the module-level assignment
/// names (`H`, `He`, `Si`, …).
///
/// # Errors
/// [`BasisLoadError::UnknownName`] when the element is absent from the file;
/// [`BasisLoadError::Parse`] when the literal does not have the expected shape.
pub fn parse_pydict(text: &str, symbol: &str, path: &str) -> Result<ParsedBasis, BasisLoadError> {
    let body = extract_assignment(text, symbol).ok_or_else(|| BasisLoadError::UnknownName {
        name: format!("{symbol} is not defined in {path}"),
    })?;
    let node = parse_list(&body).ok_or_else(|| BasisLoadError::Parse {
        file: path.to_string(),
        line: 0,
        reason: format!("could not parse the basis literal for {symbol}"),
    })?;

    let Node::List(shells) = node else {
        return Err(BasisLoadError::Parse {
            file: path.to_string(),
            line: 0,
            reason: format!("the basis literal for {symbol} is not a list"),
        });
    };

    let mut out = Vec::with_capacity(shells.len());
    for shell in shells {
        let Node::List(rows) = shell else {
            return Err(BasisLoadError::Parse {
                file: path.to_string(),
                line: 0,
                reason: format!("a shell entry for {symbol} is not a list"),
            });
        };
        let mut rows = rows.into_iter();
        let l = match rows.next() {
            Some(Node::Num(v)) => v as u8,
            _ => {
                return Err(BasisLoadError::Parse {
                    file: path.to_string(),
                    line: 0,
                    reason: format!("a shell entry for {symbol} does not start with l"),
                });
            }
        };
        let mut exponents = Vec::new();
        // `coeffs[ctr][prim]`; the contraction count is the row width minus one.
        let mut coeffs: Vec<Vec<f64>> = Vec::new();
        for row in rows {
            let Node::List(vals) = row else {
                return Err(BasisLoadError::Parse {
                    file: path.to_string(),
                    line: 0,
                    reason: format!("a primitive row for {symbol} is not a list"),
                });
            };
            let nums: Vec<f64> = vals
                .into_iter()
                .map(|n| match n {
                    Node::Num(v) => Ok(v),
                    Node::List(_) => Err(BasisLoadError::Parse {
                        file: path.to_string(),
                        line: 0,
                        reason: format!("a primitive row for {symbol} holds a nested list"),
                    }),
                })
                .collect::<Result<_, _>>()?;
            if nums.len() < 2 {
                return Err(BasisLoadError::Parse {
                    file: path.to_string(),
                    line: 0,
                    reason: format!("a primitive row for {symbol} has no coefficient"),
                });
            }
            exponents.push(nums[0]);
            if coeffs.is_empty() {
                coeffs = vec![Vec::new(); nums.len() - 1];
            } else if coeffs.len() != nums.len() - 1 {
                return Err(BasisLoadError::Parse {
                    file: path.to_string(),
                    line: 0,
                    reason: format!(
                        "ragged contraction matrix for {symbol}: {} columns then {}",
                        coeffs.len(),
                        nums.len() - 1
                    ),
                });
            }
            for (c, v) in coeffs.iter_mut().zip(&nums[1..]) {
                c.push(*v);
            }
        }
        if exponents.is_empty() {
            continue;
        }
        out.push(ShellSpec {
            l,
            exponents,
            coeffs,
        });
    }
    Ok(ParsedBasis { shells: out })
}

/// A parsed literal: either a number or a list.
enum Node {
    Num(f64),
    List(Vec<Node>),
}

/// The text of the `SYMBOL = [...]` right-hand side, comments stripped.
fn extract_assignment(text: &str, symbol: &str) -> Option<String> {
    let want = symbol.to_ascii_uppercase();
    // Strip `#` comments line by line — no string literals appear in these
    // files, so a bare `#` scan is exact.
    let stripped: String = text
        .lines()
        .map(|l| match l.find('#') {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n");

    for (i, line) in stripped.lines().enumerate() {
        let t = line.trim_start();
        let Some(eq) = t.find('=') else { continue };
        let name = t[..eq].trim();
        if !name.eq_ignore_ascii_case(&want) || !is_identifier(name) {
            continue;
        }
        // Take everything from the `=` to the point where the brackets balance.
        let rest: String = stripped.lines().skip(i).collect::<Vec<_>>().join("\n");
        let start = rest.find('=')? + 1;
        let tail = &rest[start..];
        let open = tail.find('[')?;
        let mut depth = 0i32;
        for (j, c) in tail[open..].char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(tail[open..open + j + 1].to_string());
                    }
                }
                _ => {}
            }
        }
        return None;
    }
    None
}

fn is_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Parse a balanced `[...]` literal of numbers and nested lists.
fn parse_list(s: &str) -> Option<Node> {
    let bytes: Vec<char> = s.chars().collect();
    let mut pos = 0usize;
    let node = parse_node(&bytes, &mut pos)?;
    Some(node)
}

fn parse_node(s: &[char], pos: &mut usize) -> Option<Node> {
    skip_ws(s, pos);
    if *pos >= s.len() {
        return None;
    }
    if s[*pos] == '[' {
        *pos += 1;
        let mut items = Vec::new();
        loop {
            skip_ws(s, pos);
            if *pos >= s.len() {
                return None;
            }
            if s[*pos] == ']' {
                *pos += 1;
                return Some(Node::List(items));
            }
            if s[*pos] == ',' {
                *pos += 1;
                continue;
            }
            items.push(parse_node(s, pos)?);
        }
    }
    // A number: sign, digits, `.`, exponent.
    let start = *pos;
    if s[*pos] == '+' || s[*pos] == '-' {
        *pos += 1;
    }
    while *pos < s.len()
        && (s[*pos].is_ascii_digit()
            || s[*pos] == '.'
            || s[*pos] == 'e'
            || s[*pos] == 'E'
            || ((s[*pos] == '+' || s[*pos] == '-') && matches!(s[*pos - 1], 'e' | 'E')))
    {
        *pos += 1;
    }
    if *pos == start {
        return None;
    }
    let text: String = s[start..*pos].iter().collect();
    text.parse::<f64>().ok().map(Node::Num)
}

fn skip_ws(s: &[char], pos: &mut usize) {
    while *pos < s.len() && s[*pos].is_whitespace() {
        *pos += 1;
    }
}
