"""生成自签名 SSL 证书 (用于 HTTPS, 解锁手机陀螺仪/VR模式传感器)"""
import datetime
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pathlib import Path
import socket

# 获取本机所有 IP
hostname = socket.gethostname()
try:
    lan_ip = socket.gethostbyname(hostname)
except Exception:
    lan_ip = "127.0.0.1"

print(f"主机名: {hostname}, IP: {lan_ip}")

# 生成 RSA 2048 密钥
key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

# 证书 Subject (包含主机名和 IP)
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COMMON_NAME, f"{hostname}"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "3D AI Chat"),
])

# 添加 SAN (Subject Alternative Name) 以支持 IP 访问
san_list = [
    x509.DNSName(hostname),
    x509.DNSName("localhost"),
    x509.IPAddress(__import__('ipaddress').ip_address("127.0.0.1")),
]
# 尝试添加局域网 IP
try:
    san_list.append(x509.IPAddress(__import__('ipaddress').ip_address(lan_ip)))
except Exception:
    pass

cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
    .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
    .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
    .sign(key, hashes.SHA256())
)

# 写入文件
cert_path = Path("cert.pem")
key_path = Path("key.pem")

cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
key_path.write_bytes(key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption(),
))

print(f"证书已生成: {cert_path.resolve()}, {key_path.resolve()}")
print(f"有效期: 365 天")
print(f"手机访问时需手动信任证书 (自签名)")
