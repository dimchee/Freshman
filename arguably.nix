{ buildPythonPackage
, fetchPypi
, pkgs
, poetry-core
,
}:
buildPythonPackage rec {
      pname = "arguably";
      version = "1.2.5";
      format = "pyproject";
      src = fetchPypi {
        inherit pname version;
        sha256 = "sha256-1Vljq7q4I7itHaeYIWzWpxBrMjgXiFXWrUlb7tYKtCM=";
      };
      nativeBuildInputs = [
        poetry-core
      ];
      doCheck = false;
      propagatedBuildInputs = [
        pkgs.python3Packages.docstring-parser
      ];
    }
