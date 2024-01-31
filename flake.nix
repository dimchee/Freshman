{
  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = with nixpkgs.legacyPackages.x86_64-linux; mkShell {
      buildInputs = [
        entr
        ruff-lsp
        pyright
        (python3.withPackages (ps: with ps; [
          gymnasium
          pybox2d
          pygame
          pytest
          tqdm
          decorator
          moviepy setuptools
        ])) 
        (writeShellScriptBin "watch" ''
          ls **/*.py | ${entr}/bin/entr -cc python -m freshman $@
        '')
        (writeShellScriptBin "run" ''
          python -m freshman $@
        '')
      ];
    };
  };
}
