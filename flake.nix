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
        ])) 
        (writeShellScriptBin "run" ''
          ls **/*.py | ${entr}/bin/entr sh -c 'clear && python -m rl'
        '')
      ];
    };
  };
}
