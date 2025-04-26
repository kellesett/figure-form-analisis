import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from itertools import product
from collections import defaultdict


class Model:
    def __init__(self):
        pass

    @classmethod
    def get_unique(cls, lines, rho_eps, theta_eps):
        idxs = []
        for i, line in enumerate(lines):
            if not idxs:
                idxs.append(i)
                continue
            if np.any(np.min(np.abs(lines[idxs] - lines[i]), axis=0) > (rho_eps, theta_eps)):
                idxs.append(i)
        return lines[idxs]
    
    @classmethod
    def find_rect(cls, lines, eps, alpha=1 / 3, recursion=2, root=True):
        groups_stats = []
        groups = []
        for i, line in enumerate(lines):
            if not groups:
                groups.append([line])
                groups_stats.append(line[1])
                continue
            for i, (group, stat) in enumerate(zip(groups, groups_stats)):
                if np.abs(stat - line[1]) < eps:
                    groups_stats[i] = (stat * len(group) + line[1])
                    groups[i].append(line)
                    groups_stats[i] /= len(group)
                    break
                elif np.abs(np.abs(stat - line[1]) - np.pi / 2) < eps:
                    groups[i].append(line)
                    break
            else:
                groups.append([line])
                groups_stats.append(line[1])

        if recursion:
            for i in range(len(groups)):
                cnt_parallel = 0
                cnt_ort = 0
                for line in groups[i]:
                    if np.abs(groups_stats[i] - line[1]) < eps:
                        cnt_parallel += 1
                    elif np.abs(np.abs(groups_stats[i] - line[1]) - np.pi / 2) < eps:
                        cnt_ort += 1
                if cnt_parallel  > 2 or cnt_ort > 2 or len(groups[i]) > 4:
                    new_groups = cls.find_rect(groups[i], eps * alpha, alpha=alpha, recursion=recursion-1, root=False)
                    groups[i] = new_groups[0]
                    groups_stats[i] = new_groups[0][0][1]
                    for new_group in new_groups[1:]:
                        groups.append(new_group)
                        groups_stats.append(new_group[0][1])
        if root:
            new_group = []
            for i, group in enumerate(groups):
                if len(group) < 2:
                    new_group.append(group[0])
            new_groups = cls.find_rect(new_group, eps, recursion=0, root=False)
            for new_group in new_groups:
                groups.append(new_group)
                groups_stats.append(new_group[0][1])
        return groups

    @classmethod
    def line_intersection(cls, line1, line2, eps=1e-6):
        """
        Находит точку пересечения двух линий в параметрическом виде (rho, theta).
        
        Параметры:
        - line1, line2: кортежи (rho, theta)
        - eps: погрешность для проверки параллельности
        
        Возвращает:
        - (x, y) или None (если линии параллельны)
        """
        rho1, theta1 = line1
        rho2, theta2 = line2

        cosθ1 = np.cos(theta1)
        sinθ1 = np.sin(theta1)
        cosθ2 = np.cos(theta2)
        sinθ2 = np.sin(theta2)

        denominator = cosθ1 * sinθ2 - sinθ1 * cosθ2

        # Проверка на параллельность
        if abs(denominator) < eps:
            return None

        # Вычисление координат пересечения
        x = (rho1 * sinθ2 - rho2 * sinθ1) / denominator
        y = (rho2 * cosθ1 - rho1 * cosθ2) / denominator

        return np.array((x, y))

    @classmethod
    def line_to_rho_theta(cls, x1, y1, x2, y2):
        # Вычисляем коэффициенты уравнения прямой ax + by - c = 0
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1

        # Вычисляем rho и theta
        denominator = np.sqrt(a**2 + b**2)
        if denominator == 0:
            return (0, 0)  # Точки совпадают
        
        rho = np.abs(c) / denominator
        theta = np.arctan2(b, a)
        
        # Корректируем theta в диапазон [0, π)
        if theta < 0:
            theta += 2 * np.pi
        return np.array((-rho * np.sign(c), theta))

    @classmethod
    def find_rects(cls, group):
        if len(group) == 4:
            tmp = sorted(group, key=lambda x: x[1])
            int1 = cls.line_intersection(tmp[0], tmp[2])
            int2 = cls.line_intersection(tmp[0], tmp[3])
            a = np.linalg.norm(int1 - int2)
            int3 = cls.line_intersection(tmp[1], tmp[3])
            int4 = cls.line_intersection(tmp[1], tmp[2])
            b = np.linalg.norm(int1 - int4)
        return int1, int2, int3, int4

    @classmethod
    def draw_rec(cls, img, mask, group, a, b, eps=0.1):
        box = None
        sides = None
        if len(group) < 2:
            pass
        elif len(group) > 4:
            print('Need to divide group')
        elif len(group) == 4:
            tmp = sorted(group, key=lambda x: x[1])
            if np.abs(- tmp[0][1] + tmp[3][1] - np.pi) < eps:
                tmp[1], tmp[3] = tmp[3], tmp[1]
            int1 = cls.line_intersection(tmp[0], tmp[2])
            int2 = cls.line_intersection(tmp[1], tmp[2])
            int3 = cls.line_intersection(tmp[1], tmp[3])
            int4 = cls.line_intersection(tmp[0], tmp[3])
            box = np.intp([int1, int2, int3, int4])
            cv2.drawContours(img, [box], 0, (0, 255, 0), thickness=5)

            sides = np.array(sorted([np.linalg.norm(int1 - int2), np.linalg.norm(int2 - int3)]))
        elif len(group) == 3:
            tmp = sorted(group, key=lambda x: x[1])
            int1 = cls.line_intersection(tmp[0], tmp[2])
            if np.abs(tmp[0][1] - tmp[1][1]) < eps:
                int2 = cls.line_intersection(tmp[1], tmp[2])
                direction = tmp[2]
            else:
                int2 = cls.line_intersection(tmp[0], tmp[1])
                direction = tmp[0]
            cv2.circle(img, (int(int1[0]), int(int1[1])), 5, (255, 0, 0), -1)
            cv2.circle(img, (int(int2[0]), int(int2[1])), 5, (255, 0, 0), -1)
            if np.abs(np.linalg.norm(int2 - int1) - a) < 20.:
                sz = b
            else:
                sz = a

            rho, theta = direction
            A = np.cos(theta)
            B = np.sin(theta)

            grid = [-1, 1]
            hist = []
            for i in grid:            
                x1 = int(int1[0] + i * sz * A)
                y1 = int(int1[1] + i * sz * B)

                x2 = int(int2[0] + i * sz * A)
                y2 = int(int2[1] + i * sz * B)

                box = np.intp([(int(int1[0]), int(int1[1])), (x1, y1), (x2, y2), (int(int2[0]), int(int2[1]))])
                backgroud = np.zeros_like(img)
                sample = cv2.drawContours(backgroud, [box], 0, (0, 255, 0), thickness=cv2.FILLED)
                sample = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)

                intersection = cv2.bitwise_and(mask, sample)
                score = cv2.countNonZero(intersection)
                hist.append(score)
            i = grid[np.argmax(hist)]
            x1 = int(int1[0] + i * sz * A)
            y1 = int(int1[1] + i * sz * B)
            x2 = int(int2[0] + i * sz * A)
            y2 = int(int2[1] + i * sz * B)

            box = np.intp([(int(int1[0]), int(int1[1])), (x1, y1), (x2, y2), (int(int2[0]), int(int2[1]))])
            sample = cv2.drawContours(img, [box], 0, (0, 255, 0), thickness=5)
        else:
            tmp = sorted(group, key=lambda x: x[1])
            int1 = cls.line_intersection(tmp[0], tmp[-1])

            grid = list(product([-1, 1], [-1, 1], [(a, b), (b, a)]))
            hist = []
            for i, j, rot in grid:
                rho, theta = tmp[0]
                A = np.cos(theta)
                B = np.sin(theta)
                x0 = int(int1[0])
                y0 = int(int1[1])
                x1 = int(x0 + i * rot[0] * A)
                y1 = int(y0 + i * rot[0] * B)
        
                rho, theta = tmp[-1]
                A = np.cos(theta)
                B = np.sin(theta)
                x0 = int(int1[0])
                y0 = int(int1[1])
                x2 = int(x0 + j * rot[1] * A)
                y2 = int(y0 + j * rot[1] * B)
        
                cv2.circle(img, (int(int1[0]), int(int1[1])), 5, (255, 0, 0), -1)
        
                opposite = x0 + (x1 + x2 - 2 * x0), y0 + (y1 + y2 - 2 * y0)
        
                box = np.intp([opposite, (x1, y1), (x0, y0), (x2, y2)])
                backgroud = np.zeros_like(img)
                sample = cv2.drawContours(backgroud, [box], 0, (0, 255, 0), thickness=cv2.FILLED)
                sample = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)

                intersection = cv2.bitwise_and(mask, sample)
                score = cv2.countNonZero(intersection)
                hist.append(score)
            best = grid[np.argmax(hist)]

            i, j, rot = best
            rho, theta = tmp[0]
            A = np.cos(theta)
            B = np.sin(theta)
            x0 = int(int1[0])
            y0 = int(int1[1])
            x1 = int(x0 + i * rot[0] * A)
            y1 = int(y0 + i * rot[0] * B)

            rho, theta = tmp[-1]
            A = np.cos(theta)
            B = np.sin(theta)
            x0 = int(int1[0])
            y0 = int(int1[1])
            x2 = int(x0 + j * rot[1] * A)
            y2 = int(y0 + j * rot[1] * B)

            cv2.circle(img, (int(int1[0]), int(int1[1])), 5, (255, 0, 0), -1)

            opposite = x0 + (x1 + x2 - 2 * x0), y0 + (y1 + y2 - 2 * y0)

            box = np.intp([opposite, (x1, y1), (x0, y0), (x2, y2)])
            cv2.drawContours(img, [box], 0, (0, 255, 0), thickness=5)
        return box, sides

    @classmethod
    def filter_img_bounds(cls, img, lines):
        idxs = []
        for i, line in enumerate(lines):
            if np.abs(line[0]) < 2.:
                continue
            if np.abs(line[0] - img.shape[0]) < 2. and np.abs(line[1] - np.pi / 2) < 0.01:
                continue
            if np.abs(line[0] - img.shape[1]) < 2. and np.abs(line[1]) < 0.01:
                continue
            idxs.append(i)
        return lines[idxs]

    @classmethod
    def get_approx(cls, img, cnt):
        background = np.uint8(np.zeros_like(img))
        cv2.drawContours(background, [cnt], -1, (255, 0, 0), thickness=1)
        background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
        lines = cv2.HoughLinesP(
            background,              # Бинарное изображение с краями
            rho=1,              # Разрешение ρ (в пикселях)
            theta=np.pi/1440,     # Разрешение θ (в радианах)
            threshold=30,       # Минимальное количество "голосов" для обнаружения линии
            minLineLength=25,    # Минимальная длина отрезка (в пикселях)
            maxLineGap=5
        )
        if lines is None:
            return cnt, 'S'
        
        epsilon = 0.0175 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if cv2.isContourConvex(approx):
            return approx, f'P{len(approx)}C'
        else:
            return approx, f'P{len(approx)}'

    @classmethod
    def draw_centered_text(cls, image, center, text, 
                      font_scale=0.75, thickness=2,
                      rect_color=(255, 255, 255), text_color=(255, 0, 0),
                      padding=10, font=cv2.FONT_HERSHEY_SIMPLEX):
        # Получаем размеры текста
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Рассчитываем размеры прямоугольника
        rect_width = text_width + 2 * padding
        rect_height = text_height + baseline + 2 * padding
        
        # Рассчитываем координаты прямоугольника
        x = int(center[0] - rect_width // 2)
        y = int(center[1] - rect_height // 2)
        x2 = x + rect_width
        y2 = y + rect_height
        
        # Рисуем прямоугольник
        cv2.rectangle(image, (x, y), (x2, y2), rect_color, cv2.FILLED)
        
        # Рассчитываем позицию текста
        text_x = x + padding
        text_y = y2 - padding - baseline
        
        # Рисуем текст
        cv2.putText(
            image, text, (text_x, text_y),
            font, font_scale, text_color, thickness,
            cv2.LINE_AA
        )
        return image


    def detect(self, image):
        pass
    

class CardDetector(Model):
    def __init__(self):
        super().__init__()

    
    def detect(self, img):
        figures = defaultdict(list)
        meta = {}

        mask = cv2.inRange(img, np.array([0, 0, 0]), np.array([165, 150, 150]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]

        for i, cnt in enumerate(contours):
            fig, ax = plt.subplots(1, 2, figsize=(9, 4))
            fig.set_tight_layout(True)

            background = np.zeros_like(img)
            mask = cv2.drawContours(background, [cnt], -1, (255, 0, 0), thickness=cv2.FILLED)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=5)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.erode(mask, kernel, iterations=5)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
            
            background = np.zeros_like(img)
            show_img = cv2.drawContours(background.copy(), contours, -1, (255, 0, 0), thickness=cv2.FILLED)
            show_img = cv2.cvtColor(show_img, cv2.COLOR_RGB2GRAY)
            
            contoured_img = cv2.drawContours(background, contours, -1, (255, 0, 0), thickness=1)
            contoured_img = cv2.cvtColor(contoured_img, cv2.COLOR_RGB2GRAY)
            
            ax[0].set_title(f'Выделенная компонента\nсвязности {i}')
            ax[0].imshow(show_img, cmap='Blues')
            ax[0].axis(False)
            plt.close(fig)
            
            lines = cv2.HoughLinesP(
                contoured_img,              # Бинарное изображение с краями
                rho=1,              # Разрешение ρ (в пикселях)
                theta=np.pi/360,     # Разрешение θ (в радианах)
                threshold=50,       # Минимальное количество "голосов" для обнаружения линии
                minLineLength=50,    # Минимальная длина отрезка (в пикселях)
                maxLineGap=25
            )
            lines = lines.reshape(lines.shape[0], lines.shape[2])
            lines_new = np.array([self.line_to_rho_theta(*line) for line in lines])
            lines_new = self.get_unique(lines_new, 15., 0.1)
            
            lined_img = img.copy()
            if lines is not None:
                for line, line_new in zip(lines, lines_new):
                    x1, y1, x2, y2 = line
                    cv2.line(lined_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
                    rho, theta = line_new
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 2000 * (-b))
                    y1 = int(y0 + 2000 * (a))
                    x2 = int(x0 - 2000 * (-b))
                    y2 = int(y0 - 2000 * (a))
                    cv2.line(lined_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            ax[1].set_title('Аппроксимация границ\nпрямыми линиями')
            ax[1].imshow(lined_img)
            ax[1].axis(False)
            plt.close(fig)
            figures['Показать процесс обработки компонент связности'].append(fig)
            meta['Показать процесс обработки компонент связности'] = (True, 11)

        final_img = img.copy()
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        fig.set_tight_layout(True)
        mask = cv2.inRange(img, np.array([0, 0, 0]), np.array([165, 150, 150]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
        
        background = np.zeros_like(img)

        boxes = []
        sides = np.array([96.2, 148.7])
        for i, cnt in enumerate(contours):
            mask = cv2.drawContours(background.copy(), np.array([cnt]), -1, (255, 0, 0), thickness=cv2.FILLED)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=5)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.erode(mask, kernel, iterations=5)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
            
            contoured_img = cv2.drawContours(background.copy(), contours, -1, (255, 0, 0), thickness=1)
            contoured_img = cv2.cvtColor(contoured_img, cv2.COLOR_RGB2GRAY)
            
            lines = cv2.HoughLinesP(
                contoured_img,              # Бинарное изображение с краями
                rho=1,              # Разрешение ρ (в пикселях)
                theta=np.pi/360,     # Разрешение θ (в радианах)
                threshold=50,       # Минимальное количество "голосов" для обнаружения линии
                minLineLength=50,    # Минимальная длина отрезка (в пикселях)
                maxLineGap=25
            )
            lines = lines.reshape(lines.shape[0], lines.shape[2])
            lines = np.array([self.line_to_rho_theta(*line) for line in lines])
            lines = self.filter_img_bounds(img, lines)
            lines = self.get_unique(lines, 15., 0.1)
            
            lined_img = img.copy()
            if lines is not None:
                for line in lines:
                    rho, theta = line
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 2000 * (-b))
                    y1 = int(y0 + 2000 * (a))
                    x2 = int(x0 - 2000 * (-b))
                    y2 = int(y0 - 2000 * (a))
                    cv2.line(lined_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
            groups = self.find_rect(lines, 0.15)
            for group in sorted(groups, key=lambda arr: len(arr), reverse=True):
                box, new_sides = self.draw_rec(final_img, mask, group, *sides)
                if box is not None:
                    boxes.append(box)
                if new_sides is not None:
                    sides = new_sides
        
        ax[0].set_title('Начальное изображение')
        ax[0].imshow(img)
        ax[0].axis(False)

        ax[1].set_title(f'Выделено {len(boxes)} карт')
        ax[1].imshow(final_img)
        ax[1].axis(False)
        figures['Результаты поиска карт'].append(fig)
        meta['Результаты поиска карт'] = (False, 10)
        return figures, len(boxes), meta


class BaseDetector(Model):
    def __init__(self):
        super().__init__()

    def detect(self, img):
        figures, res_cnt, meta = CardDetector().detect(img)

        edges_mask = np.zeros_like(img)
            
        doll_img = img.copy()
        mask = cv2.inRange(img, np.array([0, 0, 0]), np.array([165, 150, 150]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
        
        background = np.zeros_like(img)
        
        boxes = []
        sides = np.array([96.2, 148.7])
        for i, cnt in enumerate(contours):
            mask = cv2.drawContours(background.copy(), np.array([cnt]), -1, (255, 0, 0), thickness=cv2.FILLED)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=5)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.erode(mask, kernel, iterations=5)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
            
            contoured_img = cv2.drawContours(background.copy(), contours, -1, (255, 0, 0), thickness=1)
            contoured_img = cv2.cvtColor(contoured_img, cv2.COLOR_RGB2GRAY)
            
            lines = cv2.HoughLinesP(
                contoured_img,              # Бинарное изображение с краями
                rho=1,              # Разрешение ρ (в пикселях)
                theta=np.pi/360,     # Разрешение θ (в радианах)
                threshold=50,       # Минимальное количество "голосов" для обнаружения линии
                minLineLength=50,    # Минимальная длина отрезка (в пикселях)
                maxLineGap=25
            )
            lines = lines.reshape(lines.shape[0], lines.shape[2])
            lines = np.array([self.line_to_rho_theta(*line) for line in lines])
            lines = self.filter_img_bounds(img, lines)
            lines = self.get_unique(lines, 15., 0.1)
            
            lined_img = img.copy()
            if lines is not None:
                for line in lines:
                    rho, theta = line
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 2000 * (-b))
                    y1 = int(y0 + 2000 * (a))
                    x2 = int(x0 - 2000 * (-b))
                    y2 = int(y0 - 2000 * (a))
                    cv2.line(lined_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
            groups = self.find_rect(lines, 0.15)
            for group in sorted(groups, key=lambda arr: len(arr), reverse=True):
                final_img = img.copy()
                box, new_sides = self.draw_rec(doll_img, mask, group, *sides)
                if box is not None:
                    boxes.append(box)
                    sample = cv2.drawContours(final_img, [box], 0, (0, 255, 0), thickness=5)
                    sample = cv2.drawContours(edges_mask, [box], 0, (255, 0, 0), thickness=10)
                    if new_sides is not None:
                        sides = new_sides
        
        contoured_img = img.copy()
        polydb_img = img.copy()
        
        for box in boxes:
            center = np.sum(box, axis=0)
            
            mask = np.zeros_like(img)
            mask = cv2.drawContours(mask, [box], 0, (0, 255, 0), thickness=cv2.FILLED)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
            card_cnt = np.zeros_like(img)
            card_cnt = cv2.drawContours(card_cnt, [box], 0, (0, 255, 0), thickness=10)
            card_cnt = cv2.bitwise_not(cv2.cvtColor(card_cnt, cv2.COLOR_RGB2GRAY))
        
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            r, g, b = cv2.split(img)

            masked_img = cv2.bitwise_and(g, g, mask=mask)
        
            edges = cv2.adaptiveThreshold(
                masked_img,                      # Исходное изображение
                maxValue=255,              # Максимальное значение (белый цвет)
                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,  # Лучше для границ
                thresholdType=cv2.THRESH_BINARY,     # Светлые пиксели → белые
                blockSize=19,              # Размер окрестности (нечетный)
                C=-5                      # Корректировка порога
            )
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)
        
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.erode(edges, kernel, iterations=1)
        
            edges = cv2.bitwise_and(edges, card_cnt - np.min(card_cnt))

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)
        
            if contours:
                cv2.drawContours(contoured_img, contours[0:1], 0, (0, 255, 0), thickness=cv2.FILLED)
                approx, label = self.get_approx(img, contours[0])
                cv2.drawContours(polydb_img, [approx], 0, (0, 255, 0), thickness=3)
                self.draw_centered_text(polydb_img, np.mean(box, axis=0) + (box[0] - np.mean(box, axis=0)) / 2, label)

        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        fig.set_tight_layout(True)

        ax[0].set_title('Сегментированные фигуры')
        ax[0].imshow(contoured_img)
        ax[0].axis(False)

        ax[1].set_title('Определенные характеристики')
        ax[1].imshow(polydb_img)
        ax[1].axis(False)

        figures['Промежуточные результаты'].append(fig)
        meta['Промежуточные результаты'] = (False, 0)

        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        fig.set_tight_layout(True)

        ax.imshow(polydb_img)
        ax.axis(False)

        figures['Финальные результаты'].append(fig)
        meta['Финальные результаты'] = (False, 1)
        return figures, res_cnt, meta