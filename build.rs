use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=./cuda");

    let arch_type = env::var("ARCH_TYPE").unwrap_or(String::from("native"));
    let mut arch = String::from("-arch=");
    arch.push_str(&arch_type);

    let mut nvcc = cc::Build::new();

    nvcc.cuda(true);
    nvcc.debug(false);
    nvcc.flag(&arch);
    nvcc.files([
        "./cuda/mult.cu",
    ]);
    nvcc.compile("ingo_challenge");
}