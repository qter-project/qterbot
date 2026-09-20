{ pkgs, lib, config, inputs, ... }:

let
  pkgs-qdrant-fixed = import inputs.nixpkgs-qdrant-fixed { system = pkgs.stdenv.system; };
in {
  packages = with pkgs; [
    pkgs-qdrant-fixed.qdrant
    libz
    black
  ];
  
  languages.python = {
    enable = true;
    version = "3.14.0";
    lsp.enable = true;
    lsp.package = pkgs.python314Packages.python-lsp-server;
    venv.enable = true;
    poetry.enable = true;
    poetry.install.enable = true;
  };
}
