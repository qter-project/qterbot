{ pkgs, lib, config, inputs, ... }:

let
  pkgs-qdrant-fixed = import inputs.nixpkgs-qdrant-fixed { system = pkgs.stdenv.system; };
in {
  packages = with pkgs; [
    pkgs-qdrant-fixed.qdrant
    libz
  ];
  
  languages.python.enable = true;
  languages.python.version = "3.14.0";
  languages.python.venv.enable = true;
  languages.python.poetry.enable = true;
  languages.python.poetry.install.enable = true;
}
