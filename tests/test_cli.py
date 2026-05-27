"""Tests for the Click CLI surface."""

from click.testing import CliRunner

from qql import QQLConfig
from qql.cli import main


def test_connect_saves_no_verify_config(mocker):
    mock_client = mocker.MagicMock()
    mock_client_cls = mocker.patch("qdrant_client.QdrantClient", return_value=mock_client)
    save_config = mocker.patch("qql.cli.save_config")
    launch_repl = mocker.patch("qql.cli._launch_repl")

    result = CliRunner().invoke(
        main,
        ["connect", "--url", "https://internal.example.io", "--no-verify"],
    )

    assert result.exit_code == 0
    mock_client_cls.assert_called_once_with(
        url="https://internal.example.io", api_key=None, verify=False
    )
    save_config.assert_called_once_with(
        QQLConfig(url="https://internal.example.io", secret=None, verify=False)
    )
    launch_repl.assert_called_once()


def test_connect_saves_custom_ca_bundle_config(tmp_path, mocker):
    ca_cert = tmp_path / "internal-ca.pem"
    ca_cert.write_text("certificate")
    mock_client = mocker.MagicMock()
    mock_client_cls = mocker.patch("qdrant_client.QdrantClient", return_value=mock_client)
    save_config = mocker.patch("qql.cli.save_config")
    mocker.patch("qql.cli._launch_repl")

    result = CliRunner().invoke(
        main,
        [
            "connect",
            "--url",
            "https://internal.example.io",
            "--ca-cert",
            str(ca_cert),
        ],
    )

    assert result.exit_code == 0
    verify = str(ca_cert.resolve())
    mock_client_cls.assert_called_once_with(
        url="https://internal.example.io", api_key=None, verify=verify
    )
    save_config.assert_called_once_with(
        QQLConfig(url="https://internal.example.io", secret=None, verify=verify)
    )


def test_connect_rejects_ca_bundle_when_verification_is_disabled(tmp_path):
    ca_cert = tmp_path / "internal-ca.pem"
    ca_cert.write_text("certificate")

    result = CliRunner().invoke(
        main,
        [
            "connect",
            "--url",
            "https://internal.example.io",
            "--no-verify",
            "--ca-cert",
            str(ca_cert),
        ],
    )

    assert result.exit_code != 0
    assert "--ca-cert cannot be used with --no-verify" in result.output


def test_execute_uses_saved_verify_config(tmp_path, mocker):
    script = tmp_path / "script.qql"
    script.write_text("SHOW COLLECTIONS")
    cfg = QQLConfig(url="https://internal.example.io", secret="s3cr3t", verify=False)
    mock_client = mocker.MagicMock()
    mock_client_cls = mocker.patch("qdrant_client.QdrantClient", return_value=mock_client)
    mocker.patch("qql.cli.load_config", return_value=cfg)
    run_script = mocker.patch("qql.script.run_script", return_value=(1, 0))

    result = CliRunner().invoke(main, ["execute", str(script)])

    assert result.exit_code == 0
    mock_client_cls.assert_called_once_with(
        url="https://internal.example.io", api_key="s3cr3t", verify=False
    )
    run_script.assert_called_once()
