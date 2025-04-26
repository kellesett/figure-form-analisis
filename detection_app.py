import streamlit as st
from PIL import Image
import os
import numpy as np
from models import BaseDetector, CardDetector
import cv2


IMAGES_PATH = os.path.join('.', 'samples')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')


def detect(image, model_name):
    model = st.session_state.models[model_name]
    return model.detect(image)


def get_image_files(folder_path, level):
    names = [f for f in sorted(os.listdir(folder_path), key=lambda s: s.split('_')[1][:-4]) if f.lower().endswith(IMAGE_EXTENSIONS)]
    if level == 'Begginer & Itermediate':
        return [name for name in names if int(name.split('_')[1][:-4]) < 9]
    if level == 'Expert':
        return [name for name in names if int(name.split('_')[1][:-4]) >= 9]


def crop_center_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return img.crop((left, top, right, bottom))


def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'input'
        st.session_state.result_page_num = 0
        st.session_state.results = []
        st.session_state.selected_files = set()
        st.session_state.selected_images = []
        st.session_state.original_image = None
        st.session_state.models = {
            'CardsOnly': CardDetector(),
            'FullTask': BaseDetector(),
        }

    if st.session_state.page == 'input':
        render_input_page()
    elif st.session_state.page == 'results':
        render_results_page()

def render_input_page():
    st.title("Приложения для распознования формы объектов")
    st.subheader("Информация об алгоритмах:")
    st.markdown('\n'.join([
        '- CardsOnly - Алгоритм, реализующий сементацию карты на изображении на основе описанного в отчете геометрического подхода.',
        '- FullTask - Включает в себя первый алгоритм, а также выделяет рисунки на картах и определяет их геометрические характеристики при помощи алгоритма Рамера-Дугласа-Пекера',
    ]))

    st.sidebar.header("Параметры")

    # Model selection
    selected_model = st.sidebar.selectbox(
        "Выбрать алгоритм:",
        ["CardsOnly", "FullTask"],
    )

    # Image source selection
    image_source = st.sidebar.radio(
        "Как загрузить изображение?",
        ["Выбрать из доступных", "Загрузить свое"]
    )

    image = None
    if image_source == "Выбрать из доступных":
        selected_level = st.sidebar.selectbox(
            "Выберите уровень",
            ["Begginer & Itermediate", "Expert"],
        )

        # Title
        st.title("Выберите изображения для тестирования алгоритмов")

        # Load images
        st.session_state.level_files = []
        if st.session_state.level_files:
            image_files = st.session_state.level_files
        else:
            image_files = get_image_files(IMAGES_PATH, level=selected_level)
            st.session_state.level_files = image_files
        cols = st.columns(5)

        # Display images and clickable buttons
        for idx, image_file in enumerate(image_files):
            img_path = os.path.join(IMAGES_PATH, image_file)
            image = crop_center_square(Image.open(img_path))

            with cols[idx % 5]:  # Place image in column
                st.image(image, caption=image_file, use_container_width=True)
                if image_file not in st.session_state.selected_files:
                    if st.button(f"Выбор {image_file}", key=f"select_{idx}"):
                        st.session_state.selected_files.add(image_file)
                        st.rerun()
                else:
                    if st.button(f"Убрать {image_file}", key=f"deselect_{idx}"):
                        st.session_state.selected_files.remove(image_file)
                        st.rerun()

        st.markdown("### Выбранные изображения:")
        if st.session_state.selected_files:
            selected_list = ", ".join([f"{file}" for file in sorted(st.session_state.selected_files)])
            st.markdown(selected_list)
        else:
            st.write("Вы пока не выбрали изображения.")
    else:
        uploaded_file = st.sidebar.file_uploader("Загрузить изображение:", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            print(uploaded_file, type(uploaded_file))
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is not None:
            st.image(image, caption="Выбранное изображение", use_container_width=True)
            st.session_state.selected_images = [image]


    # Detection button
    if st.sidebar.button("Выпускайте кракена"):
        is_good = False
        if not selected_model:
            st.sidebar.error("Please select at least one model")
        if not st.session_state.selected_images:
            if not st.session_state.selected_files:
                st.sidebar.error("Please select/upload an image")
            else:
                st.session_state.selected_images = [cv2.cvtColor(cv2.imread(os.path.join(IMAGES_PATH, name)), cv2.COLOR_BGR2RGB) for name in sorted(st.session_state.selected_files)]
                is_good = True
        else:
            is_good = True
        if is_good:
            # Process image with selected models
            results = []
            for image in st.session_state.selected_images:
                figs, cnt, meta = detect(image, selected_model)
                results.append({
                    "model": selected_model,
                    "figs": figs,
                    "cnt": cnt,
                    "meta": meta,
                })
            
            st.session_state.results = results
            st.session_state.page = 'results'
            st.rerun()

def render_results_page():
    st.title("Результаты работы алгоритма")
    
    col1, col2, col3 = st.columns(3)

    with col2:
        if st.button("Назад на главную"):
            st.session_state.page = 'input'
            st.session_state.selected_images = []
            st.session_state.selected_files = set()
            st.session_state.result_page_num = 0
            st.rerun()

    with col1:
        if st.session_state.result_page_num > 0:
            if st.button("<- Предыдущий результат"):
                st.session_state.result_page_num -= 1
                st.rerun()
    
    with col3:
        if st.session_state.result_page_num < len(st.session_state.results) - 1:
            if st.button("Следующий результат ->"):
                st.session_state.result_page_num += 1
                st.rerun()

    idx = st.session_state.result_page_num
    result = st.session_state.results[idx]
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Начальное изображение")
        st.image(st.session_state.selected_images[idx], use_container_width=True)
    with col2:
        st.subheader("Результаты подсчета")
        st.metric("Найдено карт", result['cnt'])
    st.subheader(f"Визуализация преобразований для алгоритма {result['model']}")
    keys = sorted(result['meta'].keys(), key=lambda k: result['meta'][k][1])

    for title in keys:
        figs = result['figs'][title]
        if result['meta'][title][0]:
            with st.expander(f"{title}"):
                for fig in figs:
                    st.pyplot(fig, dpi=300.)
        else:
            st.subheader(f"{title}")
            for fig in figs:
                st.pyplot(fig, dpi=300.)


if __name__ == "__main__":
    main()