# WSL Ubuntu에서 Zybo Z7-20용 PetaLinux 빌드 튜토리얼

이 문서는 Windows PC에서 WSL2 Ubuntu 22.04 LTS를 설치한 뒤, Vivado에서 만든 XSA 파일을 이용해 Zynq-7020/Zybo Z7-20용 PetaLinux 이미지를 빌드하고 SD 카드에 `BOOT.BIN`, `boot.scr`, `image.ub`, `rootfs`를 저장하는 전체 과정을 정리한다.

기준 버전은 다음과 같다.

```text
Ubuntu:   22.04 LTS, WSL2
Vivado:   2022.2
Vitis:    2022.2
PetaLinux 2022.2
FPGA:     Zynq-7020, Zybo Z7-20
```

중요한 원칙은 Vivado, Vitis, PetaLinux 버전을 맞추는 것이다.

```text
Vivado 2022.2에서 export한 XSA -> PetaLinux 2022.2에서 사용
Vivado 2023.2에서 export한 XSA -> PetaLinux 2023.2에서 사용
```

서로 다른 버전을 섞으면 device tree, FSBL, bitstream, BSP 설정에서 문제가 생길 수 있다.

---

## 1. Windows에서 WSL2 설치

PowerShell을 관리자 권한으로 실행한다.

```powershell
wsl --install
```

이미 WSL이 설치되어 있다면 업데이트한다.

```powershell
wsl --update
```

WSL 상태를 확인한다.

```powershell
wsl --version
wsl -l -v
```

기본 WSL 버전을 2로 설정한다.

```powershell
wsl --set-default-version 2
```

---

## 2. Ubuntu 22.04 LTS 설치

PowerShell에서 Ubuntu 22.04를 설치한다.

```powershell
wsl --install -d Ubuntu-22.04
```

설치 후 실행한다.

```powershell
wsl -d Ubuntu-22.04
```

처음 실행하면 Linux 사용자 이름과 비밀번호를 만든다.

```text
username: 원하는 사용자 이름
password: 원하는 비밀번호
```

비밀번호 입력 중 화면에 문자가 표시되지 않는 것은 정상이다.

Ubuntu 버전을 확인한다.

```bash
lsb_release -a
```

---

## 3. Ubuntu 기본 업데이트

Ubuntu 터미널에서 실행한다.

```bash
sudo apt update
sudo apt upgrade -y
```

PetaLinux 프로젝트와 빌드는 Windows 경로인 `/mnt/c`, `/mnt/d` 아래가 아니라 WSL 내부 경로에서 하는 것을 권장한다.

권장 작업 폴더:

```bash
mkdir -p ~/workspace
mkdir -p ~/workspace/xsa
mkdir -p ~/workspace/petalinux_projects
mkdir -p ~/installers
```

권장:

```text
/home/<user>/workspace/petalinux_projects
```

비권장:

```text
/mnt/c/Users/...
/mnt/d/...
```

`/mnt/c`, `/mnt/d` 아래에서 Yocto/PetaLinux 빌드를 하면 파일 권한, 심볼릭 링크, 빌드 속도 문제를 만날 수 있다.

---

## 4. PetaLinux 필수 패키지 설치

Ubuntu에서 다음 패키지를 설치한다.

```bash
sudo apt install -y \
  gawk wget git diffstat unzip texinfo gcc build-essential chrpath socat \
  cpio python3 python3-pip python3-pexpect xz-utils debianutils iputils-ping \
  python3-git python3-jinja2 libegl1-mesa libsdl1.2-dev xterm \
  zstd liblz4-tool file locales net-tools \
  make g++ perl tar bzip2 gzip patch diffutils flex bison bc rsync \
  screen tmux nano vim dos2unix \
  libncurses5 libncurses5-dev libncursesw5 libncursesw5-dev \
  zlib1g zlib1g-dev
```

locale을 설정한다.

```bash
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8
```

터미널을 다시 열거나 아래 명령으로 적용한다.

```bash
source ~/.bashrc
```

확인:

```bash
locale
```

`LANG=en_US.UTF-8`이 보이면 된다.

PetaLinux installer가 `zlib1g:i386`을 요구할 수 있으므로 32-bit package architecture도 활성화한다.

```bash
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install -y zlib1g:i386
```

설치 중 다음 에러가 나오면 위 패키지들이 빠진 것이다.

```text
ERROR: You are missing these development libraries required by PetaLinux:

 - ncurses
 - zlib1g:i386
```

이 경우 아래 명령을 실행한 뒤 PetaLinux installer를 다시 실행한다.

```bash
sudo apt install -y libncurses5 libncurses5-dev libncursesw5 libncursesw5-dev

sudo dpkg --add-architecture i386
sudo apt update
sudo apt install -y zlib1g:i386
```

---

## 5. PetaLinux 설치

AMD/Xilinx 사이트에서 PetaLinux 2022.2 설치 파일을 다운로드한다.

예시 파일명:

```text
petalinux-v2022.2-final-installer.run
```

Windows에 다운로드했다면 WSL 내부로 복사한다.

예를 들어 Windows `D:\Downloads`에 설치 파일이 있다면 WSL에서는 다음처럼 접근한다.

```bash
cp /mnt/d/Downloads/petalinux-v2022.2-final-installer.run ~/installers/
```

실행 권한을 준다.

```bash
chmod +x ~/installers/petalinux-v2022.2-final-installer.run
```

설치 위치를 만든다.

```bash
mkdir -p ~/petalinux/2022.2
```

설치한다.

```bash
~/installers/petalinux-v2022.2-final-installer.run --dir ~/petalinux/2022.2
```

라이선스 동의 화면이 나오면 안내에 따라 동의한다.

설치 후 PetaLinux 환경을 적용한다.

```bash
source ~/petalinux/2022.2/settings.sh
```

주의: 파일명은 보통 `setting.sh`가 아니라 `settings.sh`이다.

정상 설치 확인:

```bash
which petalinux-create
which petalinux-config
which petalinux-build
petalinux-util --version
```

매번 터미널을 열 때마다 `source ~/petalinux/2022.2/settings.sh`를 실행해야 한다.

자동 적용을 원하면 `~/.bashrc` 맨 아래에 추가할 수 있다.

```bash
nano ~/.bashrc
```

추가:

```bash
source ~/petalinux/2022.2/settings.sh
```

다만 여러 버전의 PetaLinux를 사용할 가능성이 있으면 자동 source하지 않고 프로젝트마다 직접 source하는 것이 더 안전하다.

---

## 6. Vivado/Vitis 사용 방식

초기에는 다음 방식을 추천한다.

```text
Windows에 Vivado/Vitis 2022.2 설치
Windows Vivado에서 bitstream 생성
Windows Vivado에서 XSA export
XSA 파일만 WSL Ubuntu 내부로 복사
WSL Ubuntu에서 PetaLinux 빌드
```

WSL 안에 Vivado/Vitis를 설치할 수도 있지만 GUI, 용량, cable driver 문제 때문에 처음에는 Windows Vivado + WSL PetaLinux 조합이 더 단순하다.

Vivado에서 XSA를 export할 때는 bitstream을 포함한다.

```text
File -> Export -> Export Hardware
Include bitstream 체크
```

예시 XSA 파일:

```text
D:\khu\4-1\project\dev\vivado\zybo_wrapper.xsa
```

WSL에서는 다음 경로로 보인다.

```text
/mnt/d/khu/4-1/project/dev/vivado/zybo_wrapper.xsa
```

XSA 파일을 WSL 내부로 복사한다.

```bash
cp /mnt/d/khu/4-1/project/dev/vivado/pl_accel_v1/design_1_wrapper.xsa ~/workspace/xsa
```

확인:

```bash
ls -lh ~/workspace/xsa
```

---


## 7. PetaLinux 프로젝트 생성

PetaLinux 환경을 적용한다.

```bash
source ~/petalinux/2022.2/settings.sh
```

프로젝트 폴더로 이동한다.

```bash
cd ~/workspace/petalinux_projects
```

Zynq 프로젝트를 생성한다.

```bash
petalinux-create -t project --template zynq -n zybo_project
cd zybo_project
```

프로젝트 구조 예시:

```text
zybo_project/
  components/
  config.project
  project-spec/
```

---

## 8. XSA 적용 및 EXT4 rootfs 설정

XSA 파일이 들어있는 디렉터리를 지정한다.

```bash
petalinux-config --get-hw-description=~/workspace/xsa
```

설정 메뉴가 뜨면 다음 항목을 설정한다.

```text
Image Packaging Configuration
  -> Root filesystem type
      -> EXT4
```

저장 후 종료한다.

EXT4를 선택하는 이유는 SD 카드의 두 번째 EXT4 파티션에 `rootfs`를 풀어서 사용하기 위해서이다.

---

## 9. Kernel 설정

커널 설정 메뉴를 연다.

```bash
petalinux-config -c kernel
```

### 9.1 USB host, USB CDC ACM 설정

다음 메뉴로 이동한다.

```text
Device Drivers
  -> USB support
```

아래 항목들을 켠다.

```text
[*] Support for Host-side USB
[*] USB announce new devices
[*] EHCI HCD (USB 2.0) support
[*] USB Modem (CDC ACM) support
[*] ChipIdea Highspeed Dual Role Controller
[*] ChipIdea device controller
[*] ChipIdea host controller
```

ESP32, USB CDC 장치, USB serial 장치를 연결할 계획이면 `USB Modem (CDC ACM) support`가 중요하다.

### 9.2 AXI를 /dev/mem으로 직접 접근하는 경우

다음 메뉴에서 `/dev/mem` 지원을 확인한다.

```text
Device Drivers
  -> Character devices
    -> /dev/mem virtual device support
```

기본적으로 켜져 있는 경우가 많다.

관련 설정:

```text
CONFIG_DEVMEM=y
CONFIG_STRICT_DEVMEM is not set
```

---

## 10. Device Tree 설정

사용자 device tree 파일을 연다.

```bash
nano project-spec/meta-user/recipes-bsp/device-tree/files/system-user.dtsi
```

USB host와 reserved memory가 필요한 경우 다음 내용을 추가한다.

```dts
/ {
    usb_phy0: phy0 {
        compatible = "ulpi-phy";
        #phy-cells = <0>;
        reg = <0xe0002000 0x1000>;
        view-port = <0x0170>;
        drv-vbus;
    };

    reserved-memory {
        #address-cells = <1>;
        #size-cells = <1>;
        ranges;

        pl_buffers@1e000000 {
            no-map;
            reg = <0x1e000000 0x00200000>;
        };
    };
};

&usb0 {
    status = "okay";
    dr_mode = "host";
    usb-phy = <&usb_phy0>;
};
```

```

위 예시는 현재 Vivado Address Editor에서 다음 AXI-Lite 제어 포트가 보이는 경우에 맞춘 값이다.

```text
IP:                 full_pose_accel_0
Interface:          s_axi_control
Master Base Address 0x40000000
Range:              64K
Master High Address 0x4000FFFF
```

`Data_m_axi_gmem0`, `Data_m_axi_gmem1`, `Data_m_axi_gmem2`는 HLS IP가 DDR에 접근하기 위한 AXI master 포트이다. 

주의할 점:

```text
Vivado Address Editor의 AXI base address
system-user.dtsi의 reg 주소
C/Python 코드에서 mmap하는 주소
```

이 셋이 일치해야 한다.

현재 address map에서는 제어 레지스터 base address가 `0x40000000`이므로 device tree도 `reg = <0x40000000 0x10000>;`로 쓴다. 여기서 `0x10000`은 64KB range를 의미한다.

HLS IP가 DDR buffer를 읽고 쓰는 구조라면 `reserved-memory` 영역도 코드와 맞춰야 한다. 예를 들어 사용자 프로그램이 입력/출력 버퍼를 `0x1e000000`부터 2MB 영역에 두도록 작성되어 있다면 다음 설정을 유지한다.

```dts
reserved-memory {
    #address-cells = <1>;
    #size-cells = <1>;
    ranges;

    pl_buffers@1e000000 {
        no-map;
        reg = <0x1e000000 0x00200000>;
    };
};
```

이 reserved memory 주소는 Address Editor의 `HP0_DDR_LOWOCM` 범위 안에 있어야 한다. 현재 map에서는 HP 포트들이 `0x00000000`부터 `0x3fffffff`까지 1GB DDR 영역에 접근할 수 있으므로 `0x1e000000`은 범위 안에 있다.

저장:

```text
Ctrl + O
Enter
Ctrl + X
```

---

## 11. RootFS 패키지 설정

rootfs 설정 메뉴를 연다.

```bash
petalinux-config -c rootfs
```

메뉴에서 `/`를 누르면 검색할 수 있다.

### 11.1 USB, debug 관련 패키지

다음 패키지를 켠다.

```text
usbutils
util-linux
procps
coreutils
```

용도:

```text
usbutils   -> lsusb
util-linux -> stty, mount 등
procps     -> ps, top, free 등
coreutils  -> 기본 명령어 강화
```

### 11.2 네트워크, 파일 전송

SSH 서버로 `dropbear`를 켠다.

```text
dropbear
```

파일 전송이 필요하면 다음 패키지를 추가로 고려한다.

```text
openssh-scp
openssh-sftp-server
```

### 11.3 개발 도구

보드 위에서 간단한 C 프로그램을 빌드하려면 다음을 켠다.

```text
packagegroup-core-buildessential
```

### 11.4 Python

Python을 사용할 경우 다음을 켠다.

```text
python3
python3-modules
```

저장 후 종료한다.

---

## 12. 로그인 편의 설정

개발 중 root 로그인과 비밀번호 없는 로그인을 편하게 쓰기 위해 `debug-tweaks`를 추가한다.

```bash
nano project-spec/meta-user/conf/petalinuxbsp.conf
```

맨 아래에 추가한다.

```conf
EXTRA_IMAGE_FEATURES:append = " debug-tweaks"
```

새 프로젝트를 `petalinux-create`로 만들면 이 설정은 자동으로 기존 프로젝트에서 따라오지 않는다. 새 프로젝트마다 다시 확인해야 한다.

---

## 13. 빌드

프로젝트 루트에서 빌드한다.

```bash
petalinux-build
```

처음 빌드는 오래 걸린다. PC 성능과 네트워크 상태에 따라 수십 분 이상 걸릴 수 있다.

빌드 결과물은 주로 다음 위치에 생긴다.

```text
images/linux/
```

주요 파일:

```text
images/linux/zynq_fsbl.elf
images/linux/system.bit
images/linux/u-boot.elf
images/linux/boot.scr
images/linux/image.ub
images/linux/rootfs.tar.gz
```

---

## 14. BOOT.BIN 생성

Zynq-7000용 BOOT.BIN을 생성한다.

```bash
petalinux-package --boot \
  --fsbl images/linux/zynq_fsbl.elf \
  --fpga images/linux/system.bit \
  --u-boot \
  --force
```

성공하면 다음 파일이 생성된다.

```text
images/linux/BOOT.BIN
```

SD 카드 FAT32 파티션에는 보통 다음 세 파일이 들어간다.

```text
BOOT.BIN
boot.scr
image.ub
```

SD 카드 EXT4 파티션에는 `rootfs.tar.gz`를 풀어서 넣는다.

---


---

========================================================================

---

## 추가 사항
bitstream 파일이 변경되었을 경우, 다음과 같이 진행한다.

1. 하드웨어 설정 업데이트

```bash
petalinux-config --get-hw-description=~/workspace/xsa
```

2. SD 카드 부팅을 위한 RootFS Type 재확인 (선택적 검증)

Image Packaging Configuration
  -> Root filesystem type 
      -> [ ] INITRAMFS (디폴트로 돌아갔는지 확인)
      -> [*] EXT4 (이것으로 선택되어 있어야 함)


3. Device tree (system-user.dtsi) 주소 맵 매칭

```bash
nano project-spec/meta-user/recipes-bsp/device-tree/files/system-user.dtsi
```

USB host와 reserved memory가 필요한 경우 다음 내용을 추가한다.

```dts
/ {
    usb_phy0: phy0 {
        compatible = "ulpi-phy";
        #phy-cells = <0>;
        reg = <0xe0002000 0x1000>;
        view-port = <0x0170>;
        drv-vbus;
    };

    reserved-memory {
        #address-cells = <1>;
        #size-cells = <1>;
        ranges;

        pl_buffers@1e000000 {
            no-map;
            reg = <0x1e000000 0x00200000>;
        };
    };
};

&usb0 {
    status = "okay";
    dr_mode = "host";
    usb-phy = <&usb_phy0>;
};
```

4. build

```bash
petalinux-build
```
5. 부트 이미지 재설정

```bash
petalinux-package --boot \
  --fsbl images/linux/zynq_fsbl.elf \
  --fpga images/linux/system.bit \
  --u-boot \
  --force
```

---

========================================================================

---


## 15. SD 카드 파티션 구성

SD 카드는 두 파티션으로 구성한다.

```text
Partition 1: FAT32, BOOT, 500MB 이상
Partition 2: EXT4, rootfs, 나머지 용량
```

이미 SD 카드가 이 구조로 만들어져 있다면 다시 파티션을 만들 필요는 없다.

중요: 아래 작업에서 `/dev/sdX`는 예시이다. 실제 장치를 반드시 `lsblk`로 확인해야 한다. 잘못 선택하면 PC 디스크 데이터를 지울 수 있다.

---

## 16. WSL에서 SD 카드 인식시키기

WSL은 기본적으로 USB SD 카드 리더기를 바로 볼 수 없는 경우가 많다. Windows PowerShell 관리자 권한에서 `usbipd`를 사용한다.

설치:

```powershell
winget install dorssel.usbipd-win
```

장치 목록 확인:

```powershell
usbipd list
```

SD 카드 리더기의 `BUSID`를 찾는다.

예:

```text
BUSID  VID:PID    DEVICE
1-15   xxxx:xxxx  USB Mass Storage Device
```

처음 한 번 bind한다.

```powershell
usbipd bind --busid 1-15
```

WSL에 attach한다.

```powershell
usbipd attach --busid 1-15 --wsl
```

Ubuntu에서 SD 카드가 보이는지 확인한다.

```bash
lsblk
```

예를 들어 다음처럼 보일 수 있다.

```text
sdf      8:80   1  29.7G  0 disk
├─sdf1   8:81   1   512M  0 part
└─sdf2   8:82   1  29.2G  0 part
```

이 경우:

```text
/dev/sdf1 -> FAT32 BOOT 파티션
/dev/sdf2 -> EXT4 rootfs 파티션
```

---

## 17. SD 카드 마운트

마운트 폴더를 만든다.

```bash
mkdir -p ~/sd_boot ~/sd_root
```

실제 장치명을 확인한 뒤 마운트한다.

예시:

```bash
sudo mount /dev/sdf1 ~/sd_boot
sudo mount /dev/sdf2 ~/sd_root
```

마운트 확인:

```bash
df -h
```

---

## 18. BOOT 파티션에 파일 복사

PetaLinux 프로젝트 루트에서 실행한다.

```bash
cd ~/workspace/petalinux_projects/zybo_project
```

FAT32 BOOT 파티션에 파일을 복사한다.

```bash
sudo cp images/linux/BOOT.BIN ~/sd_boot/
sudo cp images/linux/boot.scr ~/sd_boot/
sudo cp images/linux/image.ub ~/sd_boot/
```

확인:

```bash
ls -lh ~/sd_boot
```

정상적으로 다음 파일이 보여야 한다.

```text
BOOT.BIN
boot.scr
image.ub
```

---

## 19. EXT4 파티션에 rootfs 저장

기존 rootfs 내용을 지운다.

```bash
sudo rm -rf ~/sd_root/*
```

`rootfs.tar.gz`를 EXT4 파티션에 압축 해제한다.

```bash
sudo tar -xpf images/linux/rootfs.tar.gz -C ~/sd_root
```

`-p` 옵션은 파일 권한을 보존하기 위해 중요하다.

확인:

```bash
ls -lh ~/sd_root
```

정상적으로 다음과 같은 Linux rootfs 디렉터리가 보여야 한다.

```text
bin
boot
dev
etc
home
lib
proc
run
sbin
sys
tmp
usr
var
```

---

## 20. Ethernet 고정 IP 설정

PC와 FPGA 보드를 LAN 케이블로 직접 연결하려면 보드의 `eth0` IP를 고정한다.

rootfs 안의 network 설정 파일을 수정한다.

```bash
sudo nano ~/sd_root/etc/network/interfaces
```

예시 내용:

```text
auto lo
iface lo inet loopback

auto eth0
iface eth0 inet static
    address 192.168.1.15
    netmask 255.255.255.0
```

저장 후 확인한다.

```bash
cat ~/sd_root/etc/network/interfaces
```

Windows PC 쪽 이더넷 IPv4 설정 예:

```text
IP address: 192.168.1.10
Subnet mask: 255.255.255.0
Gateway: 비워둠
DNS: 비워둠
```

Windows에서 설정 위치:

```text
Win + R
ncpa.cpl
이더넷 우클릭
속성
Internet Protocol Version 4 (TCP/IPv4)
다음 IP 주소 사용
```

---

## 21. SD 카드 unmount

복사가 끝나면 반드시 sync 후 unmount한다.

```bash
sync
sudo umount ~/sd_root
sudo umount ~/sd_boot
```

Windows PowerShell에서 WSL에 attach한 USB 장치를 분리하려면 다음을 사용할 수 있다.

```powershell
usbipd detach --busid 1-15
```

---

## 22. SD 카드 최종 구성

FAT32 BOOT 파티션:

```text
BOOT.BIN
boot.scr
image.ub
```

EXT4 rootfs 파티션:

```text
bin/
boot/
dev/
etc/
home/
lib/
proc/
run/
sbin/
sys/
tmp/
usr/
var/
```
---

## 23. Zybo Z7-20 부팅

보드 설정을 확인한다.

```text
Boot mode: SD boot
SD 카드 삽입
UART USB 케이블 연결
전원 연결
```

Ubuntu에서 UART 장치를 확인한다.

```bash
dmesg | grep tty
```

예를 들어 `/dev/ttyUSB1`이면 다음처럼 접속한다.

```bash
sudo screen /dev/ttyUSB1 115200
```

또는:

```bash
sudo minicom -D /dev/ttyUSB1 -b 115200
```

부팅 후 로그인:

```text
root
```

`debug-tweaks`가 적용되어 있으면 비밀번호 없이 로그인할 수 있다.

---

## 24. 부팅 후 확인 명령어

보드에서 실행한다.

```bash
uname -a
cat /proc/cmdline
mount
df -h
ip addr
```

USB 확인:

```bash
lsusb
dmesg | grep -i usb
```

USB CDC ACM 확인:

```bash
dmesg | grep ttyACM
ls /dev/ttyACM*
```

UIO 확인:

```bash
ls /dev/uio*
dmesg | grep -i uio
cat /proc/iomem
```

네트워크 확인:

```bash
ip addr show eth0
ping 192.168.1.10
```

PC에서 보드로 SSH 접속:

```bash
ssh root@192.168.1.15
```

---

## 25. 자주 나는 문제

### petalinux 명령어를 찾을 수 없음

증상:

```text
petalinux-create: command not found
```

해결:

```bash
source ~/petalinux/2022.2/settings.sh
```

### PetaLinux 설치 중 ncurses, zlib1g:i386가 없다고 나옴

증상:

```text
INFO: Checking installation environment requirements...
WARNING: This is not a supported OS
INFO: Checking free disk space
INFO: Checking installed tools
INFO: Checking installed development libraries
ERROR: You are missing these development libraries required by PetaLinux:

 - ncurses
 - zlib1g:i386
```

해결:

```bash
sudo apt install -y libncurses5 libncurses5-dev libncursesw5 libncursesw5-dev

sudo dpkg --add-architecture i386
sudo apt update
sudo apt install -y zlib1g:i386
```

그 다음 installer를 다시 실행한다.

```bash
~/installers/petalinux-v2022.2-final-installer.run --dir ~/petalinux/2022.2
```

`WARNING: This is not a supported OS`는 WSL Ubuntu 22.04에서 뜰 수 있는 경고다. `ncurses`, `zlib1g:i386` 에러를 해결하면 설치가 계속 진행되는 경우가 많다. 그래도 설치가 막히면 Ubuntu 20.04 같은 공식 지원 OS 환경이 더 안정적이다.

### XSA 적용 중 에러

확인할 것:

```text
Vivado와 PetaLinux 버전이 같은지
XSA export 시 bitstream을 포함했는지
XSA 파일을 /mnt/c, /mnt/d가 아니라 WSL 내부로 복사했는지
```

### BOOT.BIN은 있는데 부팅이 안 됨

확인할 것:

```text
SD boot jumper가 맞는지
FAT32 파티션에 BOOT.BIN, boot.scr, image.ub가 있는지
petalinux-package --boot를 다시 실행했는지
Vivado XSA에 bitstream이 포함되어 있는지
```

### rootfs로 넘어가지 못함

확인할 것:

```text
petalinux-config에서 Root filesystem type을 EXT4로 설정했는지
SD 카드 두 번째 파티션이 EXT4인지
rootfs.tar.gz를 sudo tar -xpf로 풀었는지
```

### /dev/uio0가 없음

확인할 것:

```text
kernel에서 CONFIG_UIO, CONFIG_UIO_PDRV_GENIRQ를 켰는지
system-user.dtsi에 compatible = "generic-uio";가 있는지
AXI base address가 Vivado Address Editor와 일치하는지
```

### lsusb 명령어가 없음

rootfs에서 `usbutils`를 켜지 않은 경우이다.

```bash
petalinux-config -c rootfs
```

에서 `usbutils`를 켠 뒤 다시 빌드한다.

---

## 26. 전체 명령어 요약

아래는 프로젝트 생성부터 SD 카드 복사까지의 핵심 명령어 요약이다.

```bash
source ~/petalinux/2022.2/settings.sh

mkdir -p ~/workspace/xsa
mkdir -p ~/workspace/petalinux_projects

cp /mnt/d/khu/4-1/project/dev/vivado/zybo_wrapper.xsa ~/workspace/xsa/

cd ~/workspace/petalinux_projects
petalinux-create -t project --template zynq -n zybo_project
cd zybo_project

petalinux-config --get-hw-description=~/workspace/xsa
petalinux-config -c kernel
petalinux-config -c rootfs

nano project-spec/meta-user/recipes-bsp/device-tree/files/system-user.dtsi
nano project-spec/meta-user/conf/petalinuxbsp.conf

petalinux-build

petalinux-package --boot \
  --fsbl images/linux/zynq_fsbl.elf \
  --fpga images/linux/system.bit \
  --u-boot \
  --force

mkdir -p ~/sd_boot ~/sd_root
lsblk

sudo mount /dev/sdf1 ~/sd_boot
sudo mount /dev/sdf2 ~/sd_root

sudo cp images/linux/BOOT.BIN ~/sd_boot/
sudo cp images/linux/boot.scr ~/sd_boot/
sudo cp images/linux/image.ub ~/sd_boot/

sudo rm -rf ~/sd_root/*
sudo tar -xpf images/linux/rootfs.tar.gz -C ~/sd_root

sudo nano ~/sd_root/etc/network/interfaces

sync
sudo umount ~/sd_root
sudo umount ~/sd_boot
```

`/dev/sdf1`, `/dev/sdf2`는 예시이므로 반드시 `lsblk` 결과를 보고 실제 SD 카드 장치명으로 바꿔야 한다.
