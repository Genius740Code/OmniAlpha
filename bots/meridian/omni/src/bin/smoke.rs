//! Smoke-test the WASM bot logic natively: feed it sample site requests.
use serde_json::json;

fn main() {
    // 1. Move request at the starting position.
    let req = json!({
        "type": "move",
        "start": "[B:A10-D10 R:P10-S10]",
        "moves": [16058],
        "limits": {"moveTimeMs": 5000},
        "seed": 7
    });
    let rep = omni::answer(&req);
    println!("move reply: {rep}");
    assert!(rep.get("move").and_then(|m| m.as_u64()).is_some(), "must return a move");

    // 2. Analysis request.
    let req = json!({
        "type": "analysis",
        "start": "[B:A10-D10 R:P10-S10]",
        "moves": [16058],
        "limits": {"visits": 32},
        "seed": 7
    });
    let rep = omni::answer(&req);
    println!("analysis reply: {rep}");
    let a = rep.get("analysis").expect("must return analysis");
    assert!(a.get("evaluation").and_then(|e| e.as_f64()).is_some());
    assert!(a.get("candidates").and_then(|c| c.as_array()).is_some());
    println!("smoke: OK");
}
