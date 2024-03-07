import wmi
import rsa
import os
import datetime
import base64


def _get_baseboard_sn():
    """
    gain baseboard sn
    :return: baseboard sn
    """
    c = wmi.WMI()
    for board_id in c.Win32_BaseBoard():
        # print(board_id.SerialNumber)
        return board_id.SerialNumber


def _read_license_content(pub_key: rsa.PublicKey, grant_content_load_pth, signature_load_pth):
    grant_content_b = None
    msg_signature_b = None
    with open(grant_content_load_pth, mode="rb") as f:
        grant_content_b = f.read()
    with open(signature_load_pth, mode="rb") as f:
        msg_signature_b = f.read()

    # verify license is legal, if not will raise an Exception
    try:
        rsa.verify(grant_content_b, msg_signature_b, pub_key)
    except rsa.pkcs1.VerificationError as e:
        print(e)
        raise Exception("Verification Failed")

    machine_id, exp_date = list(map(lambda x: x.split(":")[1], grant_content_b.decode(encoding="utf8").split(",")))
    exp_date = datetime.datetime.strptime(exp_date, "%Y-%m-%d")

    return machine_id, exp_date


def auth_license():
    """
    verify license is legal，if not will raise a Exception
    verify machine_id is legal
    verify exp_date
    :param license_pth: license file directory
    :param pub_key_pth: public.pem file path
    :return:
    """
    root = './license'

    grant_content_load_pth = os.path.join(root, "license")
    signature_load_pth = os.path.join(root, "license.sign")
    pub_key_pth = os.path.join(root, "public.pem")

    assert os.path.exists(grant_content_load_pth) and os.path.exists(signature_load_pth), 'License Not Found'

    with open(pub_key_pth, mode="rb") as f:
        pem_b = f.read()
        pub_key = rsa.PublicKey.load_pkcs1(pem_b)

    # verify license legal
    machine_id, exp_date = _read_license_content(pub_key, grant_content_load_pth, signature_load_pth)
    # print(machine_id)
    # print(exp_date)

    # # verify machine is legal(this computer as same as the license authorization)
    current_machine_id = _get_baseboard_sn()
    assert current_machine_id == machine_id, 'Wrong Machine ID'

    # verify date now is before exp date

    current_date = datetime.datetime.strptime(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                              "%Y-%m-%d %H:%M:%S")
    assert current_date <= exp_date, 'Expired'
    return machine_id, exp_date


def gen_application_code():
    with open("./license/public.pem", mode="rb") as f:
        pem_b = f.read()
        pub_key = rsa.PublicKey.load_pkcs1(pem_b)

    message = bytes(get_baseboard_sn(), encoding="utf8")
    crypto = rsa.encrypt(message, pub_key)
    crypto_b64 = base64.b64encode(crypto)
    crypto_b64_str = bytes.decode(crypto_b64)
    return crypto_b64_str


def get_baseboard_sn():
    """
    获取主板序列号
    :return: 主板序列号
    """
    c = wmi.WMI()
    for board_id in c.Win32_BaseBoard():
        # print(board_id.SerialNumber)
        return board_id.SerialNumber


def auth_check():
    msg = None
    exp_date = None
    try:
        _, exp_date = auth_license()
    except Exception as e:
        if str(e) == 'Verification Failed':
            msg = '许可证遭篡改。'
        elif str(e) == 'License Not Found':
            msg = '未找到许可证。'
        elif str(e) == 'Wrong Machine ID':
            msg = '非本机许可证。'
        elif str(e) == 'Expired':
            msg = '许可证已过期。'
        else:
            raise e
    return msg, exp_date
