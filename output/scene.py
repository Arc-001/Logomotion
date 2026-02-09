from manim import *

class PostfixToInfix(Scene):
    def construct(self):
        # --- Title Section (0-5s) ---
        title = Text("Postfix to Infix Conversion", color=BLUE).scale(1.2)
        title.to_edge(UP)
        
        subtitle = Text("Using a Stack-based Approach", font_size=24).next_to(title, DOWN)
        
        self.play(Write(title), run_time=2)
        self.play(FadeIn(subtitle))
        self.wait(3)

        # --- Rule Explanation (5-15s) ---
        rules_title = Text("The Rules:", color=YELLOW, font_size=32).shift(UP * 0.5)
        rule1 = Text("1. If Operand: Push to Stack", font_size=28).next_to(rules_title, DOWN, buff=0.5).to_edge(LEFT, buff=1)
        rule2 = Text("2. If Operator: Pop two, wrap in ( ), and Push", font_size=28).next_to(rule1, DOWN, buff=0.3).to_edge(LEFT, buff=1)
        
        self.play(Write(rules_title))
        self.play(FadeIn(rule1, shift=RIGHT))
        self.wait(2)
        self.play(FadeIn(rule2, shift=RIGHT))
        self.wait(4)
        
        self.play(FadeOut(rules_title), FadeOut(rule1), FadeOut(rule2), FadeOut(subtitle))

        # --- Example Setup (15-25s) ---
        expression_label = Text("Postfix Expression:", font_size=30).to_edge(LEFT).shift(UP * 2)
        # Expression: A B C * +
        tokens = ["A", "B", "C", "*", "+"]
        token_mobs = VGroup(*[Text(t, font_size=48, color=GOLD) for t in tokens]).arrange(RIGHT, buff=0.8)
        token_mobs.next_to(expression_label, RIGHT, buff=0.5)
        
        self.play(Write(expression_label))
        self.play(LaggedStart(*[Write(m) for m in token_mobs], lag_ratio=0.5), run_time=3)
        self.wait(2)

        # --- Stack Visualization (25-55s) ---
        stack_rect = Rectangle(height=4, width=3, color=WHITE).to_edge(DOWN).shift(RIGHT * 3)
        stack_label = Text("Stack", font_size=30).next_to(stack_rect, UP)
        self.play(Create(stack_rect), Write(stack_label))
        
        stack_contents = []

        def get_stack_pos(index):
            return stack_rect.get_bottom() + UP * (0.5 + index * 0.8)

        # Step 1: Push A
        self.play(Indicate(token_mobs[0]))
        a_stack = Text("A", font_size=36).move_to(get_stack_pos(0))
        self.play(TransformFromCopy(token_mobs[0], a_stack), run_time=2)
        stack_contents.append(a_stack)
        self.wait(2)

        # Step 2: Push B
        self.play(Indicate(token_mobs[1]))
        b_stack = Text("B", font_size=36).move_to(get_stack_pos(1))
        self.play(TransformFromCopy(token_mobs[1], b_stack), run_time=2)
        stack_contents.append(b_stack)
        self.wait(2)

        # Step 3: Push C
        self.play(Indicate(token_mobs[2]))
        c_stack = Text("C", font_size=36).move_to(get_stack_pos(2))
        self.play(TransformFromCopy(token_mobs[2], c_stack), run_time=2)
        stack_contents.append(c_stack)
        self.wait(2)

        # Step 4: Operator *
        self.play(Indicate(token_mobs[3], color=RED))
        self.wait(1)
        # Pop C and B
        op2 = stack_contents.pop()
        op1 = stack_contents.pop()
        self.play(op2.animate.shift(LEFT * 5), op1.animate.shift(LEFT * 7), run_time=2)
        
        # Using MathTex for expressions
        new_expr = MathTex("(B * C)", color=GREEN).scale(1.2).move_to(get_stack_pos(1))
        self.play(ReplacementTransform(VGroup(op1, op2), new_expr), run_time=2)
        stack_contents.append(new_expr)
        self.wait(3)

        # Step 5: Operator +
        self.play(Indicate(token_mobs[4], color=RED))
        self.wait(1)
        # Pop (B*C) and A
        op2_final = stack_contents.pop()
        op1_final = stack_contents.pop()
        self.play(op2_final.animate.shift(LEFT * 4 + UP * 1), op1_final.animate.shift(LEFT * 6 + UP * 1), run_time=2)
        
        final_infix = MathTex("(A + (B * C))", color=ORANGE).scale(1.5).center()
        self.play(FadeOut(stack_rect), FadeOut(stack_label), FadeOut(expression_label), FadeOut(token_mobs))
        self.play(ReplacementTransform(VGroup(op1_final, op2_final), final_infix), run_time=3)
        
        # --- Conclusion (55-60s) ---
        result_text = Text("Final Infix Expression", color=WHITE, font_size=32).next_to(final_infix, UP, buff=1)
        self.play(Write(result_text))
        self.wait(5)