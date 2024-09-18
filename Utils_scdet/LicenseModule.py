import base64
import datetime
import os
import pickle
import time
import traceback
from binascii import b2a_hex, a2b_hex

import requests
import rsa
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5, AES
from Crypto.PublicKey import RSA

from Utils.ArgumentsModule import ArgumentManager
from Utils.StaticParameters import appDir, USER_PRIVACY_GRANT_PATH, IS_DEBUG, RELEASE_TYPE

import wmi
# import uuid

if ArgumentManager.is_steam:
    import steamworks
    from steamworks import GenericSteamException


class RSACipher(object):
    private_pem = None
    public_pem = None

    def __init__(self):
        self.private_pem = b''
        self.public_pem = b''

    def get_public_key(self):
        return self.public_pem

    def get_private_key(self):
        return self.private_pem

    def decrypt_with_private_key(self, _cipher_text):
        try:
            _rsa_key = RSA.importKey(self.private_pem)
            _cipher = Cipher_pkcs1_v1_5.new(_rsa_key)
            _text = _cipher.decrypt(base64.b64decode(_cipher_text), "ERROR")
            return _text.decode(encoding="utf-8")
        except:
            return ""

    def encrypt_with_public_key(self, _text):
        _rsa_key = RSA.importKey(self.public_pem)
        _cipher = Cipher_pkcs1_v1_5.new(_rsa_key)
        _cipher_text = base64.b64encode(_cipher.encrypt(_text.encode(encoding="utf-8")))
        return _cipher_text

    # encrypt with private key & decrypt with public key is not allowed in Python
    # although it is allowed in RSA
    def encrypt_with_private_key(self, _text):
        _rsa_key = RSA.importKey(self.private_pem)
        _cipher = Cipher_pkcs1_v1_5.new(_rsa_key)
        _cipher_text = base64.b64encode(_cipher.encrypt(_text.encode(encoding="utf-8")))
        return _cipher_text

    def decrypt_with_public_key(self, _cipher_text):
        _rsa_key = RSA.importKey(self.public_pem)
        _cipher = Cipher_pkcs1_v1_5.new(_rsa_key)
        _text = _cipher.decrypt(base64.b64decode(_cipher_text), "ERROR")
        return _text.decode(encoding="utf-8")


class AESCipher(object):
    def __init__(self):
        self.key = ''.encode('utf-8')
        self.mode = AES.MODE_CBC
        self.iv = b''

    @staticmethod
    def _add_to_16(text: bytes):
        """
        如果text不足16位的倍数就用空格补足为16位
        :param text:
        :return:
        """
        if len(text) % 16:
            add = 16 - (len(text) % 16)
        else:
            add = 0
        text = text + (b'\0' * add)
        return text

    def _encrypt(self, text: bytes):
        """
        加密函数
        :return:
        """
        text = self._add_to_16(text)
        cryptos = AES.new(self.key, self.mode, self.iv)
        cipher_text = cryptos.encrypt(text)
        # 因为AES加密后的字符串不一定是ascii字符集的，输出保存可能存在问题，所以这里转为16进制字符串
        return b2a_hex(cipher_text)

    def _decrypt(self, text):
        """
        解密后，去掉补足的空格用strip() 去掉
        :return:
        """
        cryptos = AES.new(self.key, self.mode, self.iv)
        plain_text = cryptos.decrypt(a2b_hex(text))
        return plain_text


class ValidationBase:
    """Base class for Validation

    Designed to be compatible with SteamValidation

    Attributes:
        logger: logger
        _is_validate_start: boolean, indicating the env whether the env is save, should be set once instantiated
                            Retrieved by CheckValidateStart
    """

    def __init__(self, logger):
        self.logger = logger
        self._is_validate_start = False
        self._validate_error = None
        pass

    def CheckValidateStart(self):
        return self._is_validate_start

    def CheckDLC(self, pro_dlc_id: int):
        pass

    def GetStat(self, key: str, key_type: type):
        pass

    def GetAchv(self, key: str):
        pass

    def SetStat(self, key: str, value):
        pass

    def SetAchv(self, key: str, clear=False):
        pass

    def Store(self):
        pass

    def GetUserId(self):
        return ""

    def GetValidateError(self):
        return self._validate_error


class RetailValidation(ValidationBase):
    """Class for Retail Validation
    """

    def __init__(self, logger):
        super().__init__(logger)
        original_cwd = appDir
        self._rsa_worker = RSACipher()
        self._bin_path = os.path.join(appDir, 'license.dat')
        if ArgumentManager.release_type == RELEASE_TYPE.SAE:
            self._is_validate_start = True
            return
        try:
            self._is_validate_start = self._regist()  # This method has to be called in order for the wrapper to become functional!
        except Exception as e:
            # self._is_validate_start = True  # debug
            self._validate_error = e
            self.logger.error('Failed to initiate Retail License. Shutting down.')
        finally:
            os.chdir(original_cwd)

    def _GetCVolumeSerialNumber(self):
        import wmi
        c = wmi.WMI()
        for physical_disk in c.Win32_DiskDrive():
            return physical_disk.SerialNumber
        else:
            return 0

    def _GenerateRegisterBin(self):
        bin_data = {'license_data': self._rsa_worker.encrypt_with_private_key(self._GetCVolumeSerialNumber())}
        pickle.dump(bin_data, open(self._bin_path, 'wb'))

    def _ReadRegisterBin(self):
        if not os.path.exists(self._bin_path):
            self._GenerateRegisterBin()
            raise OSError("Could not find License File. The system had generated a .dat file at the root dir "
                          "of this app for license, please send this to administrator "
                          "and replace it with the one that was sent you")
        bin_data = pickle.load(open(self._bin_path, 'rb'))
        assert type(bin_data) is dict, "Type of License Data is not correct, " \
                                       "please consult administrator for further support"
        license_key = bin_data.get('license_key', "")
        return license_key

    def _regist(self):
        license_key = self._ReadRegisterBin()
        volume_serial = self._GetCVolumeSerialNumber()
        key_decrypted = self._rsa_worker.decrypt_with_private_key(license_key)
        if volume_serial != key_decrypted:
            self._GenerateRegisterBin()
            raise OSError("Wrong Register code, please check your license with your administrator")
        elif volume_serial == key_decrypted:
            return True

    def CheckDLC(self, pro_dlc_id: int):
        """All DLC Purchased as default"""
        return True

    def GetStat(self, key: str, key_type: type):
        return False

    def GetAchv(self, key: str):
        return False

    def SetStat(self, key: str, value):
        return

    def SetAchv(self, key: str, clear=False):
        return

    def Store(self):
        return False


PUBLIC_KEY = []
PRIVATE_KEY = []


class SteamValidation(ValidationBase):
    """Class for Steam Validation
    """

    def __init__(self, logger):
        super().__init__(logger)
        original_cwd = appDir
        self.steamworks = None
        self.steamworks = steamworks.STEAMWORKS(ArgumentManager.app_id)
        self.ip = "45.88.194.116"
        # self.ip = "127.0.0.1"
        self.ports = ["26582", "26583", "26584", "26585"]

        # if IS_DEBUG:
        #     self._is_validate_start = True  # debug
        #     os.chdir(original_cwd)
        #     return
        try:
            self.steamworks.initialize()  # This method has to be called in order for the wrapper to become functional!
        except Exception as e:
            self._is_validate_start = False  # debug
            self._validate_error = GenericSteamException(
                f'Failed to Validate the Software, Please Try Restarting Steam: {e}')
            self.logger.error('Failed to initiate Steam API. Shutting down.')
            os.chdir(original_cwd)
            return

        # Test Steam Connection
        self._is_validate_start = True
        if self.steamworks.UserStats.RequestCurrentStats():
            self.logger.info('Steam Stats successfully retrieved!')
        else:
            self._is_validate_start = False
            self._validate_error = GenericSteamException('Failed to get Stats Error, Please Make Sure Steam is On')
            self.logger.error('Failed to get Steam stats. Shutting down.')
        try:
            if self._is_validate_start and not ArgumentManager.release_type == RELEASE_TYPE.DEMO:  # demo does not require test
                self._is_validate_start = self._Check3pValidateStatus()
        except Exception as e:
            self._validate_error = e
            self.logger.error(f"Failed to Process 3P Validation\n{traceback.format_exc()}")
            self._is_validate_start = False
        os.chdir(original_cwd)

    @staticmethod
    def simple_rsa_encrypt(pk: str, content: str):
        rsa_public_key1 = pk.encode()
        key = rsa.PublicKey.load_pkcs1(rsa_public_key1)
        ss = rsa.encrypt(str.encode(content), key)
        re = base64.b64encode(ss).decode("utf-8")
        return re

    def _Check3pValidateStatus(self) -> bool:
        try:
            uid = str(self.steamworks.Users.GetSteamID())
        except:
            self.logger.debug("Insecure Device")
            self._validate_error = GenericSteamException(
                f'Unable to Locate Steam ID, Please Check your Internet Connection')
            return False
        device_id = ""
        try:
            # raise Exception("CPU Info Failed Test")
            c = wmi.WMI()
            for cpu in c.Win32_Processor():
                device_id = str(cpu.ProcessorId.strip())
                break
            # device_id = ":".join([uuid.uuid1().hex[-12:][i: i + 2] for i in range(0, 11, 2)])
        except:
            self.logger.error(f"Failed to locate 1st info")
            try:
                # raise Exception("BIOS Info Failed Test")
                c = wmi.WMI()
                for bios_id in c.Win32_BIOS():
                    device_id = str(bios_id.SerialNumber.strip())
                    break
            except:
                # self.logger.error(f"{traceback.format_exc()}")
                self.logger.debug("Insecure Device without Disk")
                self._validate_error = GenericSteamException(
                    f'Unsafe Device, SVFI does not support running on this machine')
                return False

        t = datetime.datetime.now(datetime.timezone.utc)
        # convert to unix, this will keep the utc timezone
        unix = t.timestamp()
        current_timestamp = int(unix)

        # region Test
        # uid = "76561199081327127"
        # device_id = "00:00:00:10"
        # current_timestamp = int(unix) + 999999999
        # endregion

        body_encrypted = self.simple_rsa_encrypt(''.join(PUBLIC_KEY), f"{uid}||{device_id}||{current_timestamp}")
        is_success = False

        retry = 4
        req_success = False
        while retry and not req_success:
            urls = [f"{self.ip}:{port}" for port in self.ports]
            url = urls[retry - 1]
            try:
                # region Test
                # if retry in [4, 3, 2]:
                #     raise Exception("Request Failed Test")
                # endregion
                res = requests.post(url=(f'http://{url}/api/'),
                                    data={'body': body_encrypted}, timeout=30)
                try:
                    content = res.json()['body']
                    req_success = True
                except:
                    self.logger.error(f"Parse Log Request Failed: {res.text}, {traceback.format_exc()}")
                    time.sleep(5)
            except:
                self.logger.error(f"Log on Request Failed: {traceback.format_exc()}")
                time.sleep(5)
            retry -= 1
        if not req_success:
            self.logger.debug(f"Log on Request TLE")
            self._validate_error = GenericSteamException(f'Log on Request TLE, Please Check Network Connection')
            return False

        try:
            header_time = base64.b64decode(content[88:]).decode()
            header_time = datetime.datetime.fromtimestamp(int(header_time), tz=datetime.timezone.utc)
            request_status = rsa.decrypt(base64.b64decode(content[:88]),
                                         rsa.PrivateKey.load_pkcs1((''.join(PRIVATE_KEY)).encode()))
        except:
            self.logger.debug("Parse Log on Pack Failed")
            self._validate_error = GenericSteamException(f'Log on Request Failed, Please Check Network Connection')
            return False

        request_status = int(request_status)
        if abs(datetime.datetime.now(tz=datetime.timezone.utc) - header_time).seconds > 3600:
            self._validate_error = GenericSteamException(
                f'Invalid Log on Request, Please Connect to Internet and flush time')
            self.logger.error(f"Failed Timing, {header_time}")
            return False
        if request_status in [0, 5]:
            is_success = True
        elif request_status == 1:  # TODO Not implemented
            self.logger.debug("Device Failed")
            self._validate_error = GenericSteamException(
                f'Device Validation Failed, Please contact with developer, SteamUID: {uid}')
        elif request_status == 2:
            self.logger.debug("Limit Failed")
            self._validate_error = GenericSteamException(
                f'Device Count exceeds limit, Please contact with the developer on SVFI Steam Forum with SteamUID: {uid}')
        elif request_status == 3:  # TODO Not Implemented
            self.logger.warning("IP Failed")
            is_success = True
        elif request_status == 4:
            self.logger.debug("Purchase Status Failed")
            self._validate_error = GenericSteamException(f'This User does not purchase the software, SteamUID: {uid}')
        else:
            self.logger.error(f"Interesting Code: {request_status}")
            self._validate_error = GenericSteamException(f'Invalid Request -> {request_status}, SteamUID: {uid}')

        return is_success

    def _CheckPurchaseStatus(self):
        steam_64id = self.steamworks.Users.GetSteamID()
        valid_response = self.steamworks.Users.GetAuthSessionTicket()
        self.logger.debug(f'Steam User Logged on as {steam_64id}, auth: {valid_response}')
        if valid_response != 0:  # Abnormal Purchase
            self._is_validate_start = False
            self._validate_error = GenericSteamException("Abnormal Start, Please Check Software's Purchase Status, "
                                                         f"Response: {valid_response}")

    def CheckDLC(self, dlc_id: int) -> bool:
        """

        :param dlc_id: DLC for SVFI, start from 0
        0: Pro
        1: MultiTask
        :return:
        """
        is_purchased = self.steamworks.Apps.IsDLCInstalled(dlc_id)
        self.logger.debug(f'Steam User Purchase DLC {dlc_id} Status: {is_purchased}')
        return is_purchased

    def GetStat(self, key: str, key_type: type):
        if key_type is int:
            return self.steamworks.UserStats.GetStatInt(key)
        elif key_type is float:
            return self.steamworks.UserStats.GetStatFloat(key)

    def GetAchv(self, key: str):
        return self.steamworks.UserStats.GetAchievement(key)

    def SetStat(self, key: str, value):
        return self.steamworks.UserStats.SetStat(key, value)

    def SetAchv(self, key: str, clear=False):
        if clear:
            return self.steamworks.UserStats.ClearAchievement(key)
        return self.steamworks.UserStats.SetAchievement(key)

    def GetUserId(self):
        return self.steamworks.Users.GetSteamID()

    def Store(self):
        return self.steamworks.UserStats.StoreStats()


class EULAWriter:
    eula_hi = """
[h2]EULA[/h2]
 
Important -- Please read carefully: Please ensure that you have read and understand all rights and limitations described in the End User License Agreement (the "Agreement").

Agreement

This Agreement is between you and SDT Core and its affiliates (the "Company"). You may use the Software and any ancillary printed materials only if you accept all conditions contained in this Agreement.

By installing or using the Software, you agree to be bound by the terms of this Agreement. If you do not agree to the terms of this agreement :(I) do not install the software, and (II) if you have purchased the software, immediately return it to the place of purchase with proof of purchase for a refund.

When you install the Software, you will be asked to preview and decide to accept or not accept all terms of this Agreement by clicking the "I Accept" button. By clicking the "I Accept" button, you acknowledge that you have read this Agreement and understand and agree to be bound by its terms and conditions.

Copyright

Software is protected by copyright laws, international treaty regulations, and other intellectual property laws and regulations. Copyright in the Software (including, but not limited to, any pictures, photos, animations, videos, music, text and small applications contained in the Software) and any printed materials attached to the Software are owned by the Company and its licensors.

Grant of license

The license and use of the Software shall be subject to this Agreement. The Company grants you a limited, personal, non-exclusive license to use the Software for the sole purpose of installing it on your computer. The Company reserves all rights not granted to you in this Agreement.

You agree that the Software can connect to the Internet and send limited, non-private data, which can be stored remotedly, to verify the availablity of the license granted by the information of your Steam account and the running device.

Authorized the use of

1. If the software is configured to run on a hard drive, you can install the software on no more than three computers.

2. You may make and retain a copy of the Software for backup and archiving, provided that the software and the copy belong to you.

3. You may permanently assign all of your rights under this Agreement, provided that you do not retain copies, transfer the Software (including all components, media, printed materials and any upgrades), and that the terms of this Agreement are accepted by the transferee.

4. You may contact with the developers on Steam to update the license to grant the usages for the Software to run on other computers.

Limit

1. You may not delete or obscure copyright, trademark or other ownership indicated in the Software or accompanying printed materials.

2. You may not decompile, modify, reverse engineer, disassemble, or reproduce [b]the executable software[/b].

3. You may not copy, lease, distribute, distribute or publicly display the Software, make derivative products of the Software (unless expressly permitted by the Editor and end users of this Agreement to modify the software or other documents attached to the Software), or develop the Software for commercial purposes.

4. You may not use a backup or archived copy of the Software for any other purpose, and may only replace the original copy if it is damaged or incomplete.

The trial version

If the software provided to you is a trial version with a limited term or quantity of use, you agree to stop using the Software after the trial period. You acknowledge and agree that the Software may contain code designed to prevent you from breaching these restrictions and that such code will remain on your computer after you delete the Software to prevent you from downloading other copies and re-using the trial period.

Editor and end user changes

If the Software allows you to make changes or create new content, including the output of the software (the "Editor"), you may use the editor to modify or optimize the Software, including creating new content (collectively, the "Changes"), subject to the following restrictions. Your changes (I) must conform to the registered full version of the software; (II) No alteration shall be made to the execution document; (III) shall not contain any content which is defamatory, libelous, illegal, or injurious to others or the public interest; (IV) Shall not contain any trademark, copyrighted content or third party proprietary content; (V) Shall not be used for commercial purposes, including, but not limited to, the sale of variations, pay-per-view or time-sharing services.

Termination 

This agreement shall remain valid until termination. You may terminate this agreement by uninstalling the software. If you breach any of the terms or conditions of this Agreement, this Agreement will automatically terminate without notice. The warranty, limitation of liability and compensation for damages in this Agreement shall survive termination of this Agreement.

Limited warranty and disclaimer

You understand and agree that your use of the Software and the media in which it is recorded is at your own risk. The software and media are released "as is." Except as required by applicable law, the Company warrants to the original purchaser of the Product that, under normal use, the software media storage media will be free of defects for a period of 30 days from the date of purchase. This warranty is void for defects caused by accident, abuse, negligence or misuse. If the software does not meet the warranty requirements, you may be reimbursed unilaterally, and if you return the defective software, you may receive a replacement product free of charge. The Company does not guarantee that the software and its operations and functions will meet your requirements, nor that the use of the software will be free from interruption or error.

To the maximum extent permitted by applicable law, we do not make any warranties other than the express warranties set forth above, including but not limited to implied warranties of merchantability, special purpose and non-infringement. Except for the express warranties stated above, the Company makes no warranties, warranties or representations as to the correctness, accuracy, reliability, generality or otherwise of the use of the Software and the results of the use of the Software. Some jurisdictions do not allow the exclusion or limitation of implied warranties, so the above exceptions and limitations may not apply to you.

Limitation of liability

In any case, the company and its employees and authorization are not any caused by the software to use or not use the software for any incidental, indirect, special, or punitive damage and other damage (including but not limited to the injury to person or property damage, profit loss, operation interruption, commercial information loss, privacy infringement, failure to perform his duties and negligence) is responsible for, Even if the company or its authorized representative has been made aware of the possibility of such injury. Some jurisdictions do not allow the exclusion of incidental or consequential injury, so the above exception may not apply to you.

In no event will the company be responsible for the costs associated with software injury in excess of what you actually paid for the software.

Other

If any term or provision of this End User License Agreement is found to be illegal, invalid or unenforceable for any reason, such term and portion shall be automatically renounced without affecting the validity and enforceability of the remaining provisions of this Agreement.

This Agreement contains all agreements between You and the software company and how to use them.
    
EULA

重要须知——请仔细阅读：请确保仔细阅读并理解《最终用户许可协议》（简称“协议”）中描述的所有权利与限制。
 
协议
本协议是您与SDT Core及其附属公司（简称“公司”）之间达成的协议。仅在您接受本协议中包含的所有条件的情况下，您方可使用软件及任何附属印刷材料。
安装或使用软件即表明，您同意接受本《协议》各项条款的约束。如果您不同意本《协议》中的条款：(i)请勿安装软件, (ii)如果您已经购买软件，请立即凭购买凭证将其退回购买处，并获得退款。
在您安装软件时，会被要求预览并通过点击“我接受”按钮决定接受或不接受本《协议》的所有条款。点击“我接受”按钮，即表明您承认已经阅读过本《协议》，并且理解并同意受其条款与条件的约束。
版权
软件受版权法、国际协约条例以及其他知识产权法和条例的保护。软件（包括但不限于软件中含有的任何图片、照片、动画、视频、音乐、文字和小型应用程序）及其附属于软件的任何印刷材料的版权均由公司及其许可者拥有。
 
许可证的授予
软件的授权与使用须遵从本《协议》。公司授予您有限的、个人的、非独占的许可证，允许您使用软件，并且以将其安装在您的电脑上为唯一目的。公司保留一切未在本《协议》中授予您的权利。

您同意软件向互联网发送有限的、可被远程储存的、基于您的Steam账户以及设备信息生成的非隐私数据以验证许可证的有效性。

授权使用
1. 如果软件配置为在一个硬盘驱动器上运行，您可以将软件安装在不多于三台电脑上。
2. 您可以制作和保留软件的一个副本用于备份和存档，条件是软件及副本归属于您。
3. 您可以通过Steam联系开发者更新许可证以允许程序在其他共不多于三台的电脑上运行。
 
限制
1. 您不得删除或掩盖软件或附属印刷材料注明的版权、商标或其他所有权。
2. 您不得对发行的软件进行反编译、修改、逆向工程或重制。
3. 您不得复制、租赁、发布、散布或公开展示软件，不得制作软件的衍生产品（除非编辑器和本协议最终用户变更部分或其他附属于软件的文件明确许可），或是以商业目的对软件进行开发。
4. 您不得将软件的备份或存档副本用作其他用途，只可在原始副本被损坏或残缺的情况下，用其替换原始副本。
 
试用版本
如果提供给您的软件为试用版，其使用期限或使用数量有限制，您同意在试用期结束后停止使用软件。您知晓并同意软件可能包含用于避免您突破这些限制的代码，并且这些代码会在您删除软件后仍保留在您的电脑上，避免您下载其他副本并重复利用试用期。
 
编辑器和最终用户变更
如果软件允许您进行修改或创建新内容（包括软件输出内容）（“编辑器”），您可以使用该编辑器修改或优化软件，包括创建新内容（统称“变更”），但必须遵守下列限制。您的变更(i)必须符合已注册的完整版软件；(ii)不得对执行文件进行改动；(iii)不得包含任何诽谤、中伤、违法、损害他人或公众利益的内容；(iv)不得包含任何商标、著作权保护内容或第三方的所有权内容；(v)不得用作商业目的，包括但不限于，出售变更内容、按次计费或分时服务。
 
终止
本协议在终止前都是有效的。您可以随时卸载软件来终止该协议。如果您违反了协议的任何条款或条件，本协议将自动终止，恕不另行通知。本协议中涉及到的保证、责任限制和损失赔偿的部分在协议终止后仍然有效。
 
有限保修及免责条款
您知道并同意因使用该软件及其记录该软件的媒体所产生的风险由您自行承担。该软件和媒体“照原样”发布。除非有适用法律规定，本公司向此产品的原始购买人保证，在正常使用的情况，该软件媒体存储介质在30天内（自购买之日算起）无缺陷。对于因意外、滥用、疏忽或误用引起的缺陷，该保证无效。如果软件没有达到保证要求，您可能会单方面获得补偿，如果您退回有缺陷的软件，您可以免费获得替换产品。本公司不保证该软件及其操作和功能达到您的要求，也不保证软件的使用不会出现中断或错误。
在适用法律许可的最大范围下，除了上述的明确保证之外，本公司不做其他任何保证，包括但不限于暗含性的适销保证、特殊用途保证及非侵权保证。除了上述的明确保证之外，本公司不对软件使用和软件使用结果在正确性、准确性、可靠性、通用性和其他方面做出保证、担保或陈述。部分司法管辖区不允许排除或限制暗含性保证，因此上面的例外和限制情况可能对您不适用。
 
责任范围
在任何情况下，本公司及其员工和授权商都不对任何由软件使用或无法使用软件而引起的任何附带、间接、特殊、偶然或惩罚性伤害以及其他伤害（包括但不限于对人身或财产的伤害，利益损失，运营中断，商业信息丢失，隐私侵犯，履行职责失败及疏忽）负责，即使公司或公司授权代表已知悉了存在这种伤害的可能性。部分司法管辖区不允许排除附带或间接伤害，因此，上述例外情况可能对您不适用。

在任何情况下，公司承担的和软件伤害相关的费用都不超过您对该软件实际支付的数额。
  
其他
如果发现此最终用户许可协议的任意条款或规定违法、无效或因某些原因无法强制执行，该条款和部分将被自动舍弃，不会影响本协议其余规定的有效性和可执行性。
本协议包含您和本软件公司之间的所有协议及其使用方法。

eula = True
"""

    def __init__(self):
        self.eula_dir = os.path.join(appDir, 'models')
        os.makedirs(self.eula_dir, exist_ok=True)
        self.eula_path = os.path.join(self.eula_dir, 'md5.svfi')

    def boom(self):
        with open(self.eula_path, 'w', encoding='utf-8') as w:
            w.write(self.eula_hi)


def write_share_infos_concerns(is_grant=True):
    if is_grant:
        grant_msg = "This User grants the application to send necessary diagnostic data to SVFI to help improve the software."
    else:
        grant_msg = "This User does not grant the application to send necessary diagnostic data to SVFI to help improve the software."
    with open(USER_PRIVACY_GRANT_PATH, 'w', encoding='utf-8') as w:
        w.write(grant_msg)
    return is_grant


def is_share_infos():
    if not os.path.exists(USER_PRIVACY_GRANT_PATH):
        return False
    else:
        with open(USER_PRIVACY_GRANT_PATH, 'r', encoding='utf-8') as r:
            if 'does not' in r.read():
                return False
            else:
                return True


if __name__ == "__main__":
    from Utils.utils import Tools

    _logger = Tools.get_logger("Test", "")
    _test = SteamValidation(logger=_logger)
    result = _test.CheckValidateStart()
    print(result)
