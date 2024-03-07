from PIL import ImageDraw, Image, ImageFont
import cv2
import numpy as np
import PIL


def rectangle_text(canvas, top, left, h, w, offset_y, offset_x, msg, color, font, outline=(255, 255, 255)):
    # 绘制矩形框+文字
    pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_canvas)
    draw.rectangle((left, top, left + w, top + h), fill=color, outline=outline)
    draw.multiline_text((left + offset_x, top + offset_y), msg, (255, 255, 255), font=font)
    # draw = ImageText(pil_canvas)
    # draw.write_text_box((left+20, top+25), msg, box_width=w-40, font_filename="./fonts/STZHONGS.TTF", font_size=35, place='center')
    # pil_canvas = draw.get_image()
    return cv2.cvtColor(np.asarray(pil_canvas), cv2.COLOR_RGB2BGR)


# 绘制圆形指示灯
def draw_light(canvas, top, left, diameter, color, outline=(255, 255, 255)):
    pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_canvas)
    draw.ellipse((left, top, left + diameter, top + diameter), fill=color, outline=outline)
    return cv2.cvtColor(np.asarray(pil_canvas), cv2.COLOR_RGB2BGR)


def action_lst(canvas, text_list, font, top, left, gap, light_diameter, light_offset_v):
    # 绘制指示灯及文字组成的动作列表
    pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_canvas)
    color = (128, 128, 128)
    for i, step in enumerate(text_list):
        draw.ellipse((left, top + light_offset_v, left + light_diameter, top + light_offset_v + light_diameter),
                     fill=color, outline=(255, 255, 255))
        draw.text((left + light_diameter + 40, top), step, (255, 255, 255), font=font)
        top += gap
    return cv2.cvtColor(np.asarray(pil_canvas), cv2.COLOR_RGB2BGR)


# 绘制圆角矩形框
# ref: https://stackoverflow.com/a/60388970
def _rect_with_rounded_corners(image, top, left, h, w, r, t, c):
    """
    :param image: PIL image, assumption: uni color filled rectangle
    :param r: radius of rounded corners
    :param t: thickness of border
    :param c: color of border
    :return: new PIL image of rectangle with rounded corners
    """

    draw = ImageDraw.Draw(image)

    # Draw four rounded corners
    draw.arc([(left, top), (left+2*r-1, top+2*r-1)], 180, 270, c, t)
    draw.arc([(left+w-2*r, top), (left+w, top+2*r-1)], 270, 0, c, t)
    draw.arc([(left+w-2*r, top+h-2*r), (left+w, top+h)], 0, 90, c, t)
    draw.arc([(left, top+h-2*r), (left+2*r-1, top+h)], 90, 180, c, t)

    # Draw four edges
    draw.line([(left+r-1, top+t/2-1), (left+w-r, top+t/2-1)], c, t)
    draw.line([(left+t/2-1, top+r-1), (left+t/2-1, top+h-r)], c, t)
    draw.line([(left+w-0.5*t, top+r-1), (left+w-0.5*t, top+h-r)], c, t)
    draw.line([(left+r-1, top+h-0.5*t), (left+w-r, top+h-0.5*t)], c, t)


# 实现PIL文本框
# modified based on: https://gist.github.com/pojda/8bf989a0556845aaf4662cd34f21d269
class ImageText(object):
    def __init__(self, filename_or_size_or_Image, mode='RGBA', background=(0, 0, 0, 0),
                 encoding='utf8'):
        if isinstance(filename_or_size_or_Image, str):
            self.filename = filename_or_size_or_Image
            self.image = Image.open(self.filename)
            self.size = self.image.size
        elif isinstance(filename_or_size_or_Image, (list, tuple)):
            self.size = filename_or_size_or_Image
            self.image = Image.new(mode, self.size, color=background)
            self.filename = None
        elif isinstance(filename_or_size_or_Image, PIL.Image.Image):
            self.image = filename_or_size_or_Image
            self.size = self.image.size
            self.filename = None
        self.draw = ImageDraw.Draw(self.image)
        self.encoding = encoding

    def save(self, filename=None):
        self.image.save(filename or self.filename)

    def show(self):
        self.image.show()

    def get_image(self):
        return self.image

    def get_font_size(self, text, font, max_width=None, max_height=None):
        if max_width is None and max_height is None:
            raise ValueError('You need to pass max_width or max_height')
        font_size = 1
        text_size = self.get_text_size(font, font_size, text)
        if (max_width is not None and text_size[0] > max_width) or \
                (max_height is not None and text_size[1] > max_height):
            raise ValueError("Text can't be filled in only (%dpx, %dpx)" % \
                             text_size)
        while True:
            if (max_width is not None and text_size[0] >= max_width) or \
                    (max_height is not None and text_size[1] >= max_height):
                return font_size - 1
            font_size += 1
            text_size = self.get_text_size(font, font_size, text)

    def write_text(self, xy, text, font_filename, font_size=11,
                   color=(0, 0, 0), max_width=None, max_height=None):
        x, y = xy
        if font_size == 'fill' and \
                (max_width is not None or max_height is not None):
            font_size = self.get_font_size(text, font_filename, max_width,
                                           max_height)
        text_size = self.get_text_size(font_filename, font_size, text)
        font = ImageFont.truetype(font_filename, font_size)
        if x == 'center':
            x = (self.size[0] - text_size[0]) / 2
        if y == 'center':
            y = (self.size[1] - text_size[1]) / 2
        self.draw.text((x, y), text, font=font, fill=color)
        return text_size

    def get_text_size(self, font_filename, font_size, text):
        font = ImageFont.truetype(font_filename, font_size)
        return font.getsize(text)

    def write_text_box(self, xy, text, box_width, font_filename,
                       font_size=11, color=(0, 0, 0), place='left',
                       justify_last_line=False, position='top',
                       line_spacing=1.0):
        x, y = xy
        lines = []
        line = []
        for word in text:
            if word == '\n':
                lines.append(line)
                line = []
                continue
            new_line = ''.join(line + [word])
            size = self.get_text_size(font_filename, font_size, new_line)
            text_height = size[1] * line_spacing
            last_line_bleed = text_height - size[1]
            if size[0] <= box_width:
                line.append(word)
            else:
                lines.append(line)
                line = [word]
        if line:
            lines.append(line)
        lines = [''.join(line) for line in lines]

        if position == 'middle':
            height = (self.size[1] - len(lines) * text_height + last_line_bleed) / 2
            height -= text_height  # the loop below will fix this height
        elif position == 'bottom':
            height = self.size[1] - len(lines) * text_height + last_line_bleed
            height -= text_height  # the loop below will fix this height
        else:
            height = y - text_height

        for index, line in enumerate(lines):
            height += text_height
            if place == 'left':
                self.write_text((x, height), line, font_filename, font_size,
                                color)
            elif place == 'right':
                total_size = self.get_text_size(font_filename, font_size, line)
                x_left = x + box_width - total_size[0]
                self.write_text((x_left, height), line, font_filename,
                                font_size, color)
            elif place == 'center':
                total_size = self.get_text_size(font_filename, font_size, line)
                x_left = int(x + ((box_width - total_size[0]) / 2))
                self.write_text((x_left, height), line, font_filename,
                                font_size, color)
            elif place == 'justify':
                words = line.split()
                if (index == len(lines) - 1 and not justify_last_line) or \
                        len(words) == 1:
                    self.write_text((x, height), line, font_filename, font_size,
                                    color)
                    continue
                line_without_spaces = ''.join(words)
                total_size = self.get_text_size(font_filename, font_size,
                                                line_without_spaces)
                space_width = (box_width - total_size[0]) / (len(words) - 1.0)
                start_x = x
                for word in words[:-1]:
                    self.write_text((start_x, height), word, font_filename,
                                    font_size, color)
                    word_size = self.get_text_size(font_filename, font_size,
                                                   word)
                    start_x += word_size[0] + space_width
                last_word_size = self.get_text_size(font_filename, font_size,
                                                    words[-1])
                last_word_x = x + box_width - last_word_size[0]
                self.write_text((last_word_x, height), words[-1], font_filename,
                                font_size, color)
        return (box_width, height - y)