{
  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = with nixpkgs.legacyPackages.x86_64-linux; mkShell {
      buildInputs = [
        ruff-lsp pyright
        (python3.withPackages (ps: with ps; [
          gymnasium
          pybox2d
          pygame
          pytest
          tqdm
          matplotlib
          (callPackage ./arguably.nix {})
        ])) 
        entr ffmpeg
        (writeShellScriptBin "watch" ''
          ls **/*.py | ${entr}/bin/entr -rcc python -m freshman $@
        '')
        (writeShellScriptBin "run" ''
          python -m freshman $@
        '')
      ];
    };
  };
}
# {
#   inputs = {
#     poetry2nix = {
#       url = "github:nix-community/poetry2nix";
#       inputs.nixpkgs.follows = "nixpkgs";
#     };
#   };
#   outputs = { self, nixpkgs, poetry2nix }:
#   let
#     pkgs = nixpkgs.legacyPackages.x86_64-linux;
#     inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication;
#   in {
#     packages.x86_64-linux.default = mkPoetryApplication { 
#       projectDir = ./.; 
#     };
#     devShells.x86_64-linux.default = with pkgs; mkShell {
#       inputsFrom = [ self.packages.x86_64-linux.default ];
#       packages = [
#         poetry entr 
#         ruff-lsp pyright
#         (writeShellScriptBin "watch" ''
#           ls **/*.py | ${entr}/bin/entr -rcc python -m freshman $@
#         '')
#         (writeShellScriptBin "run" ''
#           python -m freshman $@
#         '')
#       ];
#     };
#   };
# }
