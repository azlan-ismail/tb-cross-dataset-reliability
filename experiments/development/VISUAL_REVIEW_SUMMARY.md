# AI visual duplicate review — 24 September 2026

Reviewed all nine supplied sheets (review_000.jpg through review_008.jpg), covering pairs 0000–0034. All 35 cross-source candidate pairs have visibly distinct image content, with differences in rib/clavicle configuration, cardiac outline, diaphragm contour, and/or spatial texture distribution. Decisions did not rely on embedded dataset labels, TB diagnosis or acquisition-text differences alone.

Decision: retain both members of every reviewed pair. This is an AI assessment of 256-pixel contact-sheet views, not radiologist review or a patient-identity determination. No ambiguous same-image match was identified requiring an original-resolution follow-up in these 35 pairs. The complete 1,821-candidate list remains screening evidence; the remaining 1,786 within-source candidates were not individually visually adjudicated, and none reached the audit's 0.98 correlation priority threshold. This is not exhaustive duplicate exclusion.

The corrected pipeline groups exact file/pixel matches and same inferred patient IDs. It records dHash flags without merging them. A run-specific hashed review record is required for this v2 protocol. The original results remain unchanged.

V2 retains the original classifier settings and validation-only threshold policy. Montgomery outcomes were already inspected; subsequent results are exploratory and require a further untouched source for confirmation.
