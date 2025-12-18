from OpenSSL import crypto
import os

# Generate self-signed SSL certificate
def generate_self_signed_cert():
    # Create key pair
    k = crypto.PKey()
    k.generate_key(crypto.TYPE_RSA, 2048)
    
    # Create certificate
    cert = crypto.X509()
    cert.get_subject().C = "US"
    cert.get_subject().ST = "State"
    cert.get_subject().L = "City"
    cert.get_subject().O = "Organization"
    cert.get_subject().OU = "Org Unit"
    cert.get_subject().CN = "192.168.137.92"
    
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # Valid for 1 year
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(k)
    cert.sign(k, 'sha256')
    
    # Save certificate
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    # Save private key
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
    
    print("✅ SSL certificate generated: cert.pem and key.pem")
    print("⚠️  This is a self-signed certificate. Your browser will show a warning.")
    print("    Click 'Advanced' and 'Proceed anyway' to continue.")

if __name__ == "__main__":
    generate_self_signed_cert()
