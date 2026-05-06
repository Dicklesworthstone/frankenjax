//! Golden artifact tests for Jaxpr JSON serialization format.
//!
//! These tests freeze the JSON serialization format for Jaxpr and related types.
//! Unintentional changes to serialization format would break:
//! - Cached compiled programs
//! - Inter-process communication
//! - Persistent IR storage

use fj_core::{Atom, Equation, Jaxpr, Literal, Primitive, VarId};
use smallvec::smallvec;
use std::collections::BTreeMap;

// ======================== VarId Snapshots ========================

#[test]
fn snapshot_varid_zero() {
    let var = VarId(0);
    let json = serde_json::to_string(&var).unwrap();
    assert_eq!(json, "0");
}

#[test]
fn snapshot_varid_large() {
    let var = VarId(999999);
    let json = serde_json::to_string(&var).unwrap();
    assert_eq!(json, "999999");
}

// ======================== Atom Snapshots ========================

#[test]
fn snapshot_atom_var() {
    let atom = Atom::Var(VarId(42));
    let json = serde_json::to_string_pretty(&atom).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "Var": 42
    }
    "###);
}

#[test]
fn snapshot_atom_literal_f64() {
    let atom = Atom::Lit(Literal::from_f64(3.14159));
    let json = serde_json::to_string_pretty(&atom).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "Lit": {
        "F64Bits": 4614256650576692846
      }
    }
    "###);
}

#[test]
fn snapshot_atom_literal_i64() {
    let atom = Atom::Lit(Literal::I64(-42));
    let json = serde_json::to_string_pretty(&atom).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "Lit": {
        "I64": -42
      }
    }
    "###);
}

#[test]
fn snapshot_atom_literal_bool() {
    let atom = Atom::Lit(Literal::Bool(true));
    let json = serde_json::to_string_pretty(&atom).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "Lit": {
        "Bool": true
      }
    }
    "###);
}

#[test]
fn snapshot_atom_literal_complex() {
    let atom = Atom::Lit(Literal::from_complex128(1.0, -2.0));
    let json = serde_json::to_string_pretty(&atom).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "Lit": {
        "Complex128Bits": [
          4607182418800017408,
          13835058055282163712
        ]
      }
    }
    "###);
}

// ======================== Primitive Snapshots ========================

#[test]
fn snapshot_primitives_common() {
    assert_eq!(serde_json::to_string(&Primitive::Add).unwrap(), r#""Add""#);
    assert_eq!(serde_json::to_string(&Primitive::Sub).unwrap(), r#""Sub""#);
    assert_eq!(serde_json::to_string(&Primitive::Mul).unwrap(), r#""Mul""#);
    assert_eq!(serde_json::to_string(&Primitive::Div).unwrap(), r#""Div""#);
    assert_eq!(serde_json::to_string(&Primitive::Neg).unwrap(), r#""Neg""#);
    assert_eq!(serde_json::to_string(&Primitive::Exp).unwrap(), r#""Exp""#);
    assert_eq!(serde_json::to_string(&Primitive::Log).unwrap(), r#""Log""#);
    assert_eq!(serde_json::to_string(&Primitive::Sin).unwrap(), r#""Sin""#);
    assert_eq!(serde_json::to_string(&Primitive::Cos).unwrap(), r#""Cos""#);
}

#[test]
fn snapshot_primitives_linalg() {
    assert_eq!(serde_json::to_string(&Primitive::Dot).unwrap(), r#""Dot""#);
    assert_eq!(serde_json::to_string(&Primitive::Qr).unwrap(), r#""Qr""#);
    assert_eq!(serde_json::to_string(&Primitive::Svd).unwrap(), r#""Svd""#);
    assert_eq!(serde_json::to_string(&Primitive::Eigh).unwrap(), r#""Eigh""#);
    assert_eq!(serde_json::to_string(&Primitive::Cholesky).unwrap(), r#""Cholesky""#);
}

#[test]
fn snapshot_primitives_control_flow() {
    assert_eq!(serde_json::to_string(&Primitive::Cond).unwrap(), r#""Cond""#);
    assert_eq!(serde_json::to_string(&Primitive::While).unwrap(), r#""While""#);
    assert_eq!(serde_json::to_string(&Primitive::Scan).unwrap(), r#""Scan""#);
    assert_eq!(serde_json::to_string(&Primitive::Switch).unwrap(), r#""Switch""#);
}

// ======================== Jaxpr Snapshots ========================

#[test]
fn snapshot_jaxpr_identity() {
    let jaxpr = Jaxpr::new(vec![VarId(0)], vec![], vec![VarId(0)], vec![]);
    let json = serde_json::to_string_pretty(&jaxpr).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "invars": [
        0
      ],
      "constvars": [],
      "outvars": [
        0
      ],
      "equations": []
    }
    "###);
}

#[test]
fn snapshot_jaxpr_single_add() {
    let jaxpr = Jaxpr::new(
        vec![VarId(0), VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        }],
    );
    let json = serde_json::to_string_pretty(&jaxpr).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "invars": [
        0,
        1
      ],
      "constvars": [],
      "outvars": [
        2
      ],
      "equations": [
        {
          "primitive": "Add",
          "inputs": [
            {
              "Var": 0
            },
            {
              "Var": 1
            }
          ],
          "outputs": [
            2
          ],
          "params": {}
        }
      ]
    }
    "###);
}

#[test]
fn snapshot_jaxpr_with_literal() {
    let jaxpr = Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(1)],
        vec![Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(0)), Atom::Lit(Literal::from_f64(1.0))],
            outputs: smallvec![VarId(1)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        }],
    );
    let json = serde_json::to_string_pretty(&jaxpr).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "invars": [
        0
      ],
      "constvars": [],
      "outvars": [
        1
      ],
      "equations": [
        {
          "primitive": "Add",
          "inputs": [
            {
              "Var": 0
            },
            {
              "Lit": {
                "F64Bits": 4607182418800017408
              }
            }
          ],
          "outputs": [
            1
          ],
          "params": {}
        }
      ]
    }
    "###);
}

#[test]
fn snapshot_jaxpr_with_params() {
    let mut params = BTreeMap::new();
    params.insert("axes".to_string(), "0,1".to_string());

    let jaxpr = Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(1)],
        vec![Equation {
            primitive: Primitive::ReduceSum,
            inputs: smallvec![Atom::Var(VarId(0))],
            outputs: smallvec![VarId(1)],
            params,
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        }],
    );
    let json = serde_json::to_string_pretty(&jaxpr).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "invars": [
        0
      ],
      "constvars": [],
      "outvars": [
        1
      ],
      "equations": [
        {
          "primitive": "ReduceSum",
          "inputs": [
            {
              "Var": 0
            }
          ],
          "outputs": [
            1
          ],
          "params": {
            "axes": "0,1"
          }
        }
      ]
    }
    "###);
}

#[test]
fn snapshot_jaxpr_with_constvars() {
    let jaxpr = Jaxpr::new(
        vec![VarId(0)],
        vec![VarId(100)],
        vec![VarId(1)],
        vec![Equation {
            primitive: Primitive::Mul,
            inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(100))],
            outputs: smallvec![VarId(1)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        }],
    );
    let json = serde_json::to_string_pretty(&jaxpr).unwrap();
    insta::assert_snapshot!(json, @r###"
    {
      "invars": [
        0
      ],
      "constvars": [
        100
      ],
      "outvars": [
        1
      ],
      "equations": [
        {
          "primitive": "Mul",
          "inputs": [
            {
              "Var": 0
            },
            {
              "Var": 100
            }
          ],
          "outputs": [
            1
          ],
          "params": {}
        }
      ]
    }
    "###);
}
