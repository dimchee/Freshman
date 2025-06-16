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
