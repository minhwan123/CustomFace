"""CLI(main.py)에서 터미널 입력으로 수정할 부위와 눈 회전 여부를 입력받는 모듈."""


def get_user_modifications():
    parts = {
        "눈": ["left_eye", "right_eye"],
        "코": ["nose"],
        "입": ["mouth"],
        "대두": ["face_shape"]
    }

    print("어느 부위를 수정할까요? (복수 선택, 쉼표로 구분)")
    print("선택 가능: 눈, 코, 입, 대두")

    while True:
        selected_parts = input("예: 눈,코\n> ").replace(" ", "").split(",")
        valid = True
        for part in selected_parts:
            if part not in parts:
                print(f"⚠️ '{part}' 은(는) 선택할 수 없습니다. 다시 입력해주세요.")
                valid = False
                break
        if valid:
            break

    modifications = {}
    rotate_eyes = False
    for part in selected_parts:
        if part == "눈":
            while True:
                rotate = input("눈을 회전시키겠습니까? (y/n): ").strip()
                if rotate in ["y", "Y", "n", "N"]:
                    if rotate in ["y", "Y"]:
                        rotate_eyes = True
                    break
                else:
                    print("입력 오류. 'y', 'Y', 'n', 'N' 중 하나로 입력해주세요.")
        # scale은 region별 고정값만 사용
        for region_key in parts[part]:
            modifications[region_key] = True  # 선택된 부위만 True로 표시

    return modifications, rotate_eyes