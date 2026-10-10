mod searcher;
mod session;

fn print_usage() {
    println!("usage: rune-uci --model <path.rune> [--depth <n>]");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut depth = 3usize;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model" && i + 1 < args.len() {
            model = args[i + 1].clone();
            i += 1;
        } else if args[i] == "--depth" && i + 1 < args.len() {
            depth = args[i + 1].parse::<usize>().unwrap_or(3).clamp(1, 64);
            i += 1;
        } else if args[i] == "--help" {
            print_usage();
            return;
        }
        i += 1;
    }
    if model.is_empty() {
        print_usage();
        std::process::exit(1);
    }
    match session::Session::new(&model, depth) {
        Ok(mut s) => s.serve(),
        Err(e) => {
            eprintln!("rune-uci: load failed: {e}");
            std::process::exit(1);
        }
    }
}
