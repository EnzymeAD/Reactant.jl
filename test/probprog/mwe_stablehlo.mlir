module @mwe attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
func.func @main(%arg0: tensor<2xui64> {tf.aliasing_output = 1 : i32}, %arg1: tensor<2x1xf64>, %arg2: tensor<f64>, %arg3: tensor<2x2xf64>) -> (tensor<2x1xf64>, tensor<2xui64>) {
    %cst = stablehlo.constant dense<-5.000000e-01> : tensor<1x2xf64>
    %cst_16 = stablehlo.constant dense<5.000000e-01> : tensor<1x2xf64>
    %c_17 = stablehlo.constant dense<8> : tensor<3xui32>
    %c_18 = stablehlo.constant dense<19> : tensor<3xui32>
    %c_19 = stablehlo.constant dense<[0, 1, 2]> : tensor<3xui32>
    %c_22 = stablehlo.constant dense<8> : tensor<2xui32>
    %c_23 = stablehlo.constant dense<19> : tensor<2xui32>
    %c_24 = stablehlo.constant dense<[0, 1]> : tensor<2xui32>
    %c_25 = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
    %c_26 = stablehlo.constant dense<12> : tensor<ui64>
    %c_32 = stablehlo.constant dense<5> : tensor<3xui32>
    %c_33 = stablehlo.constant dense<4> : tensor<3xui32>
    %c_34 = stablehlo.constant dense<3> : tensor<3xui32>
    %c_35 = stablehlo.constant dense<2> : tensor<3xui32>
    %c_36 = stablehlo.constant dense<24> : tensor<3xui32>
    %c_37 = stablehlo.constant dense<16> : tensor<3xui32>
    %c_38 = stablehlo.constant dense<29> : tensor<3xui32>
    %c_39 = stablehlo.constant dense<17> : tensor<3xui32>
    %c_40 = stablehlo.constant dense<1> : tensor<3xui32>
    %c_41 = stablehlo.constant dense<6> : tensor<3xui32>
    %c_42 = stablehlo.constant dense<26> : tensor<3xui32>
    %c_43 = stablehlo.constant dense<15> : tensor<3xui32>
    %c_44 = stablehlo.constant dense<13> : tensor<3xui32>
    %c_45 = stablehlo.constant dense<466688986> : tensor<3xui32>
    %c_46 = stablehlo.constant dense<5> : tensor<2xui32>
    %c_47 = stablehlo.constant dense<4> : tensor<2xui32>
    %c_48 = stablehlo.constant dense<3> : tensor<2xui32>
    %c_49 = stablehlo.constant dense<2> : tensor<2xui32>
    %c_50 = stablehlo.constant dense<24> : tensor<2xui32>
    %c_51 = stablehlo.constant dense<16> : tensor<2xui32>
    %c_52 = stablehlo.constant dense<29> : tensor<2xui32>
    %c_53 = stablehlo.constant dense<17> : tensor<2xui32>
    %c_54 = stablehlo.constant dense<1> : tensor<2xui32>
    %c_55 = stablehlo.constant dense<6> : tensor<2xui32>
    %c_56 = stablehlo.constant dense<26> : tensor<2xui32>
    %c_57 = stablehlo.constant dense<15> : tensor<2xui32>
    %c_58 = stablehlo.constant dense<13> : tensor<2xui32>
    %c_59 = stablehlo.constant dense<466688986> : tensor<2xui32>
    %cst_60 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %cst_61 = stablehlo.constant dense<0.000000e+00> : tensor<3x2xf64>
    %c_62 = stablehlo.constant dense<3> : tensor<i64>
    %c_63 = stablehlo.constant dense<false> : tensor<i1>
    %cst_64 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %cst_65 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c_66 = stablehlo.constant dense<1> : tensor<i64>
    %c_67 = stablehlo.constant dense<0> : tensor<i64>
    %cst_68 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_71 = stablehlo.constant dense<1.000000e+03> : tensor<f64>
// Init constants for outer while
%init_1x2 = stablehlo.constant dense<0.0> : tensor<1x2xf64>
%init_f64 = stablehlo.constant dense<0.0> : tensor<f64>
%init_i1 = stablehlo.constant dense<false> : tensor<i1>
%6 = stablehlo.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
%814 = stablehlo.constant dense<0.0> : tensor<f64>
%1067 = stablehlo.constant dense<0> : tensor<2xui64>
%1094 = stablehlo.constant dense<0.0> : tensor<f64>

    %1096:17 = stablehlo.while(%iterArg = %init_1x2, %iterArg_72 = %init_1x2, %iterArg_73 = %init_1x2, %iterArg_74 = %init_1x2, %iterArg_75 = %init_1x2, %iterArg_76 = %init_1x2, %iterArg_77 = %init_1x2, %iterArg_78 = %init_1x2, %iterArg_79 = %init_f64, %iterArg_80 = %init_f64, %iterArg_81 = %c_67, %iterArg_82 = %init_f64, %iterArg_83 = %init_i1, %iterArg_84 = %init_i1, %iterArg_85 = %init_f64, %iterArg_86 = %init_1x2, %iterArg_87 = %arg0) : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<1x2xf64>, tensor<2xui64>
    cond {
      %1098 = stablehlo.compare  LT, %iterArg_81, %c_62,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %1099 = stablehlo.not %iterArg_83 : tensor<i1>
      %1100 = stablehlo.not %iterArg_84 : tensor<i1>
      %1101 = stablehlo.and %1098, %1099 : tensor<i1>
      %1102 = stablehlo.and %1101, %1100 : tensor<i1>
      stablehlo.return %1102 : tensor<i1>
    } do {
      // RNG splitting (replaces Threefry key splits)
      %1350, %tmp_outer_a = stablehlo.rng_bit_generator %iterArg_87, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %output_state_88, %output_89 = stablehlo.rng_bit_generator %1350, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %1365 = stablehlo.shift_right_logical %output_89, %c_26 : tensor<ui64>
      %1366 = stablehlo.or %1365, %c_25 : tensor<ui64>
      %1367 = stablehlo.bitcast_convert %1366 : (tensor<ui64>) -> tensor<f64>
      %1368 = stablehlo.subtract %1367, %cst_68 : tensor<f64>
      %1369 = stablehlo.compare  LT, %1368, %cst_64,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %1622, %tmp_outer_b = stablehlo.rng_bit_generator %output_state_88, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %1629, %tmp_outer_c = stablehlo.rng_bit_generator %1622, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %1630 = stablehlo.shift_left %c_66, %iterArg_81 : tensor<i64>
      %1631:21 = stablehlo.while(%iterArg_92 = %iterArg, %iterArg_93 = %iterArg_72, %iterArg_94 = %iterArg_73, %iterArg_95 = %iterArg_74, %iterArg_96 = %iterArg_75, %iterArg_97 = %iterArg_76, %iterArg_98 = %iterArg_77, %iterArg_99 = %iterArg_78, %iterArg_100 = %iterArg_79, %iterArg_101 = %iterArg_80, %iterArg_102 = %iterArg_81, %iterArg_103 = %iterArg_82, %iterArg_104 = %iterArg_83, %iterArg_105 = %iterArg_84, %iterArg_106 = %iterArg_85, %iterArg_107 = %c_67, %iterArg_108 = %iterArg_86, %iterArg_109 = %1622, %iterArg_110 = %cst_61, %iterArg_111 = %cst_61, %iterArg_112 = %c_67) : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>, tensor<2xui64>, tensor<3x2xf64>, tensor<3x2xf64>, tensor<i64>
      cond {
        %1677 = stablehlo.compare  LT, %iterArg_107, %1630,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %1678 = stablehlo.not %iterArg_104 : tensor<i1>
        %1679 = stablehlo.not %iterArg_105 : tensor<i1>
        %1680 = stablehlo.and %1677, %1678 : tensor<i1>
        %1681 = stablehlo.and %1680, %1679 : tensor<i1>
        stablehlo.return %1681 : tensor<i1>
      } do {
        %1677 = stablehlo.select %1369, %iterArg_95, %iterArg_92 : tensor<i1>, tensor<1x2xf64>
        %1678 = stablehlo.select %1369, %iterArg_96, %iterArg_93 : tensor<i1>, tensor<1x2xf64>
        %1679 = stablehlo.select %1369, %iterArg_97, %iterArg_94 : tensor<i1>, tensor<1x2xf64>
        %1680 = stablehlo.bitcast_convert %iterArg_109 : (tensor<2xui64>) -> tensor<2x2xui32>
        %1681 = stablehlo.reshape %1680 : (tensor<2x2xui32>) -> tensor<4xui32>
        %1682 = stablehlo.slice %1681 [0:1] : (tensor<4xui32>) -> tensor<1xui32>
        %1683 = stablehlo.slice %1681 [1:2] : (tensor<4xui32>) -> tensor<1xui32>
        %1684 = stablehlo.slice %1681 [2:3] : (tensor<4xui32>) -> tensor<1xui32>
        %1685 = stablehlo.slice %1681 [3:4] : (tensor<4xui32>) -> tensor<1xui32>
        %1686 = stablehlo.broadcast_in_dim %1682, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %1687 = stablehlo.broadcast_in_dim %1683, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %1688 = stablehlo.broadcast_in_dim %1684, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %1689 = stablehlo.broadcast_in_dim %1685, dims = [0] : (tensor<1xui32>) -> tensor<2xui32>
        %1690 = stablehlo.xor %1686, %1687 : tensor<2xui32>
        %1691 = stablehlo.xor %1690, %c_59 : tensor<2xui32>
        %1692 = stablehlo.add %c_24, %1687 : tensor<2xui32>
        %1693 = stablehlo.add %1686, %1692 : tensor<2xui32>
        %1694 = stablehlo.shift_left %1692, %c_58 : tensor<2xui32>
        %1695 = stablehlo.shift_right_logical %1692, %c_23 : tensor<2xui32>
        %1696 = stablehlo.or %1694, %1695 : tensor<2xui32>
        %1697 = stablehlo.xor %1693, %1696 : tensor<2xui32>
        %1698 = stablehlo.add %1693, %1697 : tensor<2xui32>
        %1699 = stablehlo.shift_left %1697, %c_57 : tensor<2xui32>
        %1700 = stablehlo.shift_right_logical %1697, %c_53 : tensor<2xui32>
        %1701 = stablehlo.or %1699, %1700 : tensor<2xui32>
        %1702 = stablehlo.xor %1698, %1701 : tensor<2xui32>
        %1703 = stablehlo.add %1698, %1702 : tensor<2xui32>
        %1704 = stablehlo.shift_left %1702, %c_56 : tensor<2xui32>
        %1705 = stablehlo.shift_right_logical %1702, %c_55 : tensor<2xui32>
        %1706 = stablehlo.or %1704, %1705 : tensor<2xui32>
        %1707 = stablehlo.xor %1703, %1706 : tensor<2xui32>
        %1708 = stablehlo.add %1703, %1707 : tensor<2xui32>
        %1709 = stablehlo.shift_left %1707, %c_55 : tensor<2xui32>
        %1710 = stablehlo.shift_right_logical %1707, %c_56 : tensor<2xui32>
        %1711 = stablehlo.or %1709, %1710 : tensor<2xui32>
        %1712 = stablehlo.xor %1708, %1711 : tensor<2xui32>
        %1713 = stablehlo.add %1708, %1687 : tensor<2xui32>
        %1714 = stablehlo.add %1712, %1691 : tensor<2xui32>
        %1715 = stablehlo.add %1714, %c_54 : tensor<2xui32>
        %1716 = stablehlo.add %1713, %1715 : tensor<2xui32>
        %1717 = stablehlo.shift_left %1715, %c_53 : tensor<2xui32>
        %1718 = stablehlo.shift_right_logical %1715, %c_57 : tensor<2xui32>
        %1719 = stablehlo.or %1717, %1718 : tensor<2xui32>
        %1720 = stablehlo.xor %1716, %1719 : tensor<2xui32>
        %1721 = stablehlo.add %1716, %1720 : tensor<2xui32>
        %1722 = stablehlo.shift_left %1720, %c_52 : tensor<2xui32>
        %1723 = stablehlo.shift_right_logical %1720, %c_48 : tensor<2xui32>
        %1724 = stablehlo.or %1722, %1723 : tensor<2xui32>
        %1725 = stablehlo.xor %1721, %1724 : tensor<2xui32>
        %1726 = stablehlo.add %1721, %1725 : tensor<2xui32>
        %1727 = stablehlo.shift_left %1725, %c_51 : tensor<2xui32>
        %1728 = stablehlo.shift_right_logical %1725, %c_51 : tensor<2xui32>
        %1729 = stablehlo.or %1727, %1728 : tensor<2xui32>
        %1730 = stablehlo.xor %1726, %1729 : tensor<2xui32>
        %1731 = stablehlo.add %1726, %1730 : tensor<2xui32>
        %1732 = stablehlo.shift_left %1730, %c_50 : tensor<2xui32>
        %1733 = stablehlo.shift_right_logical %1730, %c_22 : tensor<2xui32>
        %1734 = stablehlo.or %1732, %1733 : tensor<2xui32>
        %1735 = stablehlo.xor %1731, %1734 : tensor<2xui32>
        %1736 = stablehlo.add %1731, %1691 : tensor<2xui32>
        %1737 = stablehlo.add %1735, %1686 : tensor<2xui32>
        %1738 = stablehlo.add %1737, %c_49 : tensor<2xui32>
        %1739 = stablehlo.add %1736, %1738 : tensor<2xui32>
        %1740 = stablehlo.shift_left %1738, %c_58 : tensor<2xui32>
        %1741 = stablehlo.shift_right_logical %1738, %c_23 : tensor<2xui32>
        %1742 = stablehlo.or %1740, %1741 : tensor<2xui32>
        %1743 = stablehlo.xor %1739, %1742 : tensor<2xui32>
        %1744 = stablehlo.add %1739, %1743 : tensor<2xui32>
        %1745 = stablehlo.shift_left %1743, %c_57 : tensor<2xui32>
        %1746 = stablehlo.shift_right_logical %1743, %c_53 : tensor<2xui32>
        %1747 = stablehlo.or %1745, %1746 : tensor<2xui32>
        %1748 = stablehlo.xor %1744, %1747 : tensor<2xui32>
        %1749 = stablehlo.add %1744, %1748 : tensor<2xui32>
        %1750 = stablehlo.shift_left %1748, %c_56 : tensor<2xui32>
        %1751 = stablehlo.shift_right_logical %1748, %c_55 : tensor<2xui32>
        %1752 = stablehlo.or %1750, %1751 : tensor<2xui32>
        %1753 = stablehlo.xor %1749, %1752 : tensor<2xui32>
        %1754 = stablehlo.add %1749, %1753 : tensor<2xui32>
        %1755 = stablehlo.shift_left %1753, %c_55 : tensor<2xui32>
        %1756 = stablehlo.shift_right_logical %1753, %c_56 : tensor<2xui32>
        %1757 = stablehlo.or %1755, %1756 : tensor<2xui32>
        %1758 = stablehlo.xor %1754, %1757 : tensor<2xui32>
        %1759 = stablehlo.add %1754, %1686 : tensor<2xui32>
        %1760 = stablehlo.add %1758, %1687 : tensor<2xui32>
        %1761 = stablehlo.add %1760, %c_48 : tensor<2xui32>
        %1762 = stablehlo.add %1759, %1761 : tensor<2xui32>
        %1763 = stablehlo.shift_left %1761, %c_53 : tensor<2xui32>
        %1764 = stablehlo.shift_right_logical %1761, %c_57 : tensor<2xui32>
        %1765 = stablehlo.or %1763, %1764 : tensor<2xui32>
        %1766 = stablehlo.xor %1762, %1765 : tensor<2xui32>
        %1767 = stablehlo.add %1762, %1766 : tensor<2xui32>
        %1768 = stablehlo.shift_left %1766, %c_52 : tensor<2xui32>
        %1769 = stablehlo.shift_right_logical %1766, %c_48 : tensor<2xui32>
        %1770 = stablehlo.or %1768, %1769 : tensor<2xui32>
        %1771 = stablehlo.xor %1767, %1770 : tensor<2xui32>
        %1772 = stablehlo.add %1767, %1771 : tensor<2xui32>
        %1773 = stablehlo.shift_left %1771, %c_51 : tensor<2xui32>
        %1774 = stablehlo.shift_right_logical %1771, %c_51 : tensor<2xui32>
        %1775 = stablehlo.or %1773, %1774 : tensor<2xui32>
        %1776 = stablehlo.xor %1772, %1775 : tensor<2xui32>
        %1777 = stablehlo.add %1772, %1776 : tensor<2xui32>
        %1778 = stablehlo.shift_left %1776, %c_50 : tensor<2xui32>
        %1779 = stablehlo.shift_right_logical %1776, %c_22 : tensor<2xui32>
        %1780 = stablehlo.or %1778, %1779 : tensor<2xui32>
        %1781 = stablehlo.xor %1777, %1780 : tensor<2xui32>
        %1782 = stablehlo.add %1777, %1687 : tensor<2xui32>
        %1783 = stablehlo.add %1781, %1691 : tensor<2xui32>
        %1784 = stablehlo.add %1783, %c_47 : tensor<2xui32>
        %1785 = stablehlo.add %1782, %1784 : tensor<2xui32>
        %1786 = stablehlo.shift_left %1784, %c_58 : tensor<2xui32>
        %1787 = stablehlo.shift_right_logical %1784, %c_23 : tensor<2xui32>
        %1788 = stablehlo.or %1786, %1787 : tensor<2xui32>
        %1789 = stablehlo.xor %1785, %1788 : tensor<2xui32>
        %1790 = stablehlo.add %1785, %1789 : tensor<2xui32>
        %1791 = stablehlo.shift_left %1789, %c_57 : tensor<2xui32>
        %1792 = stablehlo.shift_right_logical %1789, %c_53 : tensor<2xui32>
        %1793 = stablehlo.or %1791, %1792 : tensor<2xui32>
        %1794 = stablehlo.xor %1790, %1793 : tensor<2xui32>
        %1795 = stablehlo.add %1790, %1794 : tensor<2xui32>
        %1796 = stablehlo.shift_left %1794, %c_56 : tensor<2xui32>
        %1797 = stablehlo.shift_right_logical %1794, %c_55 : tensor<2xui32>
        %1798 = stablehlo.or %1796, %1797 : tensor<2xui32>
        %1799 = stablehlo.xor %1795, %1798 : tensor<2xui32>
        %1800 = stablehlo.add %1795, %1799 : tensor<2xui32>
        %1801 = stablehlo.shift_left %1799, %c_55 : tensor<2xui32>
        %1802 = stablehlo.shift_right_logical %1799, %c_56 : tensor<2xui32>
        %1803 = stablehlo.or %1801, %1802 : tensor<2xui32>
        %1804 = stablehlo.xor %1800, %1803 : tensor<2xui32>
        %1805 = stablehlo.add %1800, %1691 : tensor<2xui32>
        %1806 = stablehlo.add %1804, %1686 : tensor<2xui32>
        %1807 = stablehlo.add %1806, %c_46 : tensor<2xui32>
        %1808 = stablehlo.xor %1688, %1689 : tensor<2xui32>
        %1809 = stablehlo.xor %1808, %c_59 : tensor<2xui32>
        %1810 = stablehlo.add %c_24, %1689 : tensor<2xui32>
        %1811 = stablehlo.add %1688, %1810 : tensor<2xui32>
        %1812 = stablehlo.shift_left %1810, %c_58 : tensor<2xui32>
        %1813 = stablehlo.shift_right_logical %1810, %c_23 : tensor<2xui32>
        %1814 = stablehlo.or %1812, %1813 : tensor<2xui32>
        %1815 = stablehlo.xor %1811, %1814 : tensor<2xui32>
        %1816 = stablehlo.add %1811, %1815 : tensor<2xui32>
        %1817 = stablehlo.shift_left %1815, %c_57 : tensor<2xui32>
        %1818 = stablehlo.shift_right_logical %1815, %c_53 : tensor<2xui32>
        %1819 = stablehlo.or %1817, %1818 : tensor<2xui32>
        %1820 = stablehlo.xor %1816, %1819 : tensor<2xui32>
        %1821 = stablehlo.add %1816, %1820 : tensor<2xui32>
        %1822 = stablehlo.shift_left %1820, %c_56 : tensor<2xui32>
        %1823 = stablehlo.shift_right_logical %1820, %c_55 : tensor<2xui32>
        %1824 = stablehlo.or %1822, %1823 : tensor<2xui32>
        %1825 = stablehlo.xor %1821, %1824 : tensor<2xui32>
        %1826 = stablehlo.add %1821, %1825 : tensor<2xui32>
        %1827 = stablehlo.shift_left %1825, %c_55 : tensor<2xui32>
        %1828 = stablehlo.shift_right_logical %1825, %c_56 : tensor<2xui32>
        %1829 = stablehlo.or %1827, %1828 : tensor<2xui32>
        %1830 = stablehlo.xor %1826, %1829 : tensor<2xui32>
        %1831 = stablehlo.add %1826, %1689 : tensor<2xui32>
        %1832 = stablehlo.add %1830, %1809 : tensor<2xui32>
        %1833 = stablehlo.add %1832, %c_54 : tensor<2xui32>
        %1834 = stablehlo.add %1831, %1833 : tensor<2xui32>
        %1835 = stablehlo.shift_left %1833, %c_53 : tensor<2xui32>
        %1836 = stablehlo.shift_right_logical %1833, %c_57 : tensor<2xui32>
        %1837 = stablehlo.or %1835, %1836 : tensor<2xui32>
        %1838 = stablehlo.xor %1834, %1837 : tensor<2xui32>
        %1839 = stablehlo.add %1834, %1838 : tensor<2xui32>
        %1840 = stablehlo.shift_left %1838, %c_52 : tensor<2xui32>
        %1841 = stablehlo.shift_right_logical %1838, %c_48 : tensor<2xui32>
        %1842 = stablehlo.or %1840, %1841 : tensor<2xui32>
        %1843 = stablehlo.xor %1839, %1842 : tensor<2xui32>
        %1844 = stablehlo.add %1839, %1843 : tensor<2xui32>
        %1845 = stablehlo.shift_left %1843, %c_51 : tensor<2xui32>
        %1846 = stablehlo.shift_right_logical %1843, %c_51 : tensor<2xui32>
        %1847 = stablehlo.or %1845, %1846 : tensor<2xui32>
        %1848 = stablehlo.xor %1844, %1847 : tensor<2xui32>
        %1849 = stablehlo.add %1844, %1848 : tensor<2xui32>
        %1850 = stablehlo.shift_left %1848, %c_50 : tensor<2xui32>
        %1851 = stablehlo.shift_right_logical %1848, %c_22 : tensor<2xui32>
        %1852 = stablehlo.or %1850, %1851 : tensor<2xui32>
        %1853 = stablehlo.xor %1849, %1852 : tensor<2xui32>
        %1854 = stablehlo.add %1849, %1809 : tensor<2xui32>
        %1855 = stablehlo.add %1853, %1688 : tensor<2xui32>
        %1856 = stablehlo.add %1855, %c_49 : tensor<2xui32>
        %1857 = stablehlo.add %1854, %1856 : tensor<2xui32>
        %1858 = stablehlo.shift_left %1856, %c_58 : tensor<2xui32>
        %1859 = stablehlo.shift_right_logical %1856, %c_23 : tensor<2xui32>
        %1860 = stablehlo.or %1858, %1859 : tensor<2xui32>
        %1861 = stablehlo.xor %1857, %1860 : tensor<2xui32>
        %1862 = stablehlo.add %1857, %1861 : tensor<2xui32>
        %1863 = stablehlo.shift_left %1861, %c_57 : tensor<2xui32>
        %1864 = stablehlo.shift_right_logical %1861, %c_53 : tensor<2xui32>
        %1865 = stablehlo.or %1863, %1864 : tensor<2xui32>
        %1866 = stablehlo.xor %1862, %1865 : tensor<2xui32>
        %1867 = stablehlo.add %1862, %1866 : tensor<2xui32>
        %1868 = stablehlo.shift_left %1866, %c_56 : tensor<2xui32>
        %1869 = stablehlo.shift_right_logical %1866, %c_55 : tensor<2xui32>
        %1870 = stablehlo.or %1868, %1869 : tensor<2xui32>
        %1871 = stablehlo.xor %1867, %1870 : tensor<2xui32>
        %1872 = stablehlo.add %1867, %1871 : tensor<2xui32>
        %1873 = stablehlo.shift_left %1871, %c_55 : tensor<2xui32>
        %1874 = stablehlo.shift_right_logical %1871, %c_56 : tensor<2xui32>
        %1875 = stablehlo.or %1873, %1874 : tensor<2xui32>
        %1876 = stablehlo.xor %1872, %1875 : tensor<2xui32>
        %1877 = stablehlo.add %1872, %1688 : tensor<2xui32>
        %1878 = stablehlo.add %1876, %1689 : tensor<2xui32>
        %1879 = stablehlo.add %1878, %c_48 : tensor<2xui32>
        %1880 = stablehlo.add %1877, %1879 : tensor<2xui32>
        %1881 = stablehlo.shift_left %1879, %c_53 : tensor<2xui32>
        %1882 = stablehlo.shift_right_logical %1879, %c_57 : tensor<2xui32>
        %1883 = stablehlo.or %1881, %1882 : tensor<2xui32>
        %1884 = stablehlo.xor %1880, %1883 : tensor<2xui32>
        %1885 = stablehlo.add %1880, %1884 : tensor<2xui32>
        %1886 = stablehlo.shift_left %1884, %c_52 : tensor<2xui32>
        %1887 = stablehlo.shift_right_logical %1884, %c_48 : tensor<2xui32>
        %1888 = stablehlo.or %1886, %1887 : tensor<2xui32>
        %1889 = stablehlo.xor %1885, %1888 : tensor<2xui32>
        %1890 = stablehlo.add %1885, %1889 : tensor<2xui32>
        %1891 = stablehlo.shift_left %1889, %c_51 : tensor<2xui32>
        %1892 = stablehlo.shift_right_logical %1889, %c_51 : tensor<2xui32>
        %1893 = stablehlo.or %1891, %1892 : tensor<2xui32>
        %1894 = stablehlo.xor %1890, %1893 : tensor<2xui32>
        %1895 = stablehlo.add %1890, %1894 : tensor<2xui32>
        %1896 = stablehlo.shift_left %1894, %c_50 : tensor<2xui32>
        %1897 = stablehlo.shift_right_logical %1894, %c_22 : tensor<2xui32>
        %1898 = stablehlo.or %1896, %1897 : tensor<2xui32>
        %1899 = stablehlo.xor %1895, %1898 : tensor<2xui32>
        %1900 = stablehlo.add %1895, %1689 : tensor<2xui32>
        %1901 = stablehlo.add %1899, %1809 : tensor<2xui32>
        %1902 = stablehlo.add %1901, %c_47 : tensor<2xui32>
        %1903 = stablehlo.add %1900, %1902 : tensor<2xui32>
        %1904 = stablehlo.shift_left %1902, %c_58 : tensor<2xui32>
        %1905 = stablehlo.shift_right_logical %1902, %c_23 : tensor<2xui32>
        %1906 = stablehlo.or %1904, %1905 : tensor<2xui32>
        %1907 = stablehlo.xor %1903, %1906 : tensor<2xui32>
        %1908 = stablehlo.add %1903, %1907 : tensor<2xui32>
        %1909 = stablehlo.shift_left %1907, %c_57 : tensor<2xui32>
        %1910 = stablehlo.shift_right_logical %1907, %c_53 : tensor<2xui32>
        %1911 = stablehlo.or %1909, %1910 : tensor<2xui32>
        %1912 = stablehlo.xor %1908, %1911 : tensor<2xui32>
        %1913 = stablehlo.add %1908, %1912 : tensor<2xui32>
        %1914 = stablehlo.shift_left %1912, %c_56 : tensor<2xui32>
        %1915 = stablehlo.shift_right_logical %1912, %c_55 : tensor<2xui32>
        %1916 = stablehlo.or %1914, %1915 : tensor<2xui32>
        %1917 = stablehlo.xor %1913, %1916 : tensor<2xui32>
        %1918 = stablehlo.add %1913, %1917 : tensor<2xui32>
        %1919 = stablehlo.shift_left %1917, %c_55 : tensor<2xui32>
        %1920 = stablehlo.shift_right_logical %1917, %c_56 : tensor<2xui32>
        %1921 = stablehlo.or %1919, %1920 : tensor<2xui32>
        %1922 = stablehlo.xor %1918, %1921 : tensor<2xui32>
        %1923 = stablehlo.add %1918, %1809 : tensor<2xui32>
        %1924 = stablehlo.add %1922, %1688 : tensor<2xui32>
        %1925 = stablehlo.add %1924, %c_46 : tensor<2xui32>
        %1926 = stablehlo.slice %1805 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %1927 = stablehlo.slice %1807 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %1928 = stablehlo.slice %1923 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %1929 = stablehlo.slice %1925 [0:1] : (tensor<2xui32>) -> tensor<1xui32>
        %1930 = stablehlo.concatenate %1926, %1927, %1928, %1929, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %1931 = stablehlo.reshape %1930 : (tensor<4xui32>) -> tensor<2x2xui32>
        %1932 = stablehlo.bitcast_convert %1931 : (tensor<2x2xui32>) -> tensor<2xui64>
        %1933 = stablehlo.slice %1805 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %1934 = stablehlo.slice %1807 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %1935 = stablehlo.slice %1923 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %1936 = stablehlo.slice %1925 [1:2] : (tensor<2xui32>) -> tensor<1xui32>
        %1937 = stablehlo.concatenate %1933, %1934, %1935, %1936, dim = 0 : (tensor<1xui32>, tensor<1xui32>, tensor<1xui32>, tensor<1xui32>) -> tensor<4xui32>
        %1938 = stablehlo.reshape %1937 : (tensor<4xui32>) -> tensor<2x2xui32>
        %1939 = stablehlo.bitcast_convert %1938 : (tensor<2x2xui32>) -> tensor<2xui64>
        %1940 = stablehlo.select %1369, %arg2, %814 : tensor<i1>, tensor<f64>
        %1941 = stablehlo.broadcast_in_dim %1940, dims = [] : (tensor<f64>) -> tensor<1x2xf64>
        %1942 = stablehlo.multiply %1940, %cst_64 : tensor<f64>
        %1943 = stablehlo.broadcast_in_dim %1942, dims = [] : (tensor<f64>) -> tensor<1x2xf64>
        %1944 = stablehlo.multiply %1943, %1679 : tensor<1x2xf64>
        %1945 = stablehlo.subtract %1678, %1944 : tensor<1x2xf64>
        %1946 = stablehlo.dot_general %1945, %arg3, contracting_dims = [1] x [1] : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
        %1947 = stablehlo.multiply %1941, %1946 : tensor<1x2xf64>
        %1948 = stablehlo.add %1677, %1947 : tensor<1x2xf64>
        %1949 = stablehlo.dot_general %1948, %1948, contracting_dims = [0, 1] x [0, 1] : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
        %1950 = stablehlo.multiply %cst_64, %1949 : tensor<f64>
        %1951 = stablehlo.add %1948, %1948 : tensor<1x2xf64>
        %1952 = stablehlo.multiply %cst_16, %1951 : tensor<1x2xf64>
        %1953 = stablehlo.multiply %1943, %1952 : tensor<1x2xf64>
        %1954 = stablehlo.subtract %1945, %1953 : tensor<1x2xf64>
        %1955 = stablehlo.dot_general %1954, %arg3, contracting_dims = [1] x [1] : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
        %1956 = stablehlo.dot_general %1954, %1955, contracting_dims = [0, 1] x [0, 1] : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
        %1957 = stablehlo.add %1949, %1956 : tensor<f64>
        %1958 = stablehlo.multiply %cst_64, %1957 : tensor<f64>
        %1959 = stablehlo.subtract %1957, %1094 : tensor<f64>
        %1960 = stablehlo.multiply %cst_64, %1959 : tensor<f64>
        %1961 = stablehlo.compare  NE, %1960, %1960,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
        %1962 = stablehlo.select %1961, %cst_60, %1960 : tensor<i1>, tensor<f64>
        %1963 = stablehlo.negate %1962 : tensor<f64>
        %1964 = stablehlo.compare  GT, %1962, %cst_71,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
        %1965 = stablehlo.exponential %1963 : tensor<f64>
        %1966 = stablehlo.minimum %1965, %cst_68 : tensor<f64>
        %1967 = stablehlo.compare  EQ, %iterArg_107, %c_67,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %1968:16 = "stablehlo.if"(%1967) ({
          stablehlo.return %1948, %1954, %1952, %1948, %1954, %1952, %1948, %1952, %1950, %1958, %c_67, %1963, %1964, %1966, %c_66, %1954 : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>
        }, {
          %1983 = stablehlo.select %1369, %iterArg_92, %1948 : tensor<i1>, tensor<1x2xf64>
          %1984 = stablehlo.select %1369, %iterArg_93, %1954 : tensor<i1>, tensor<1x2xf64>
          %1985 = stablehlo.select %1369, %iterArg_94, %1952 : tensor<i1>, tensor<1x2xf64>
          %1986 = stablehlo.select %1369, %1948, %iterArg_95 : tensor<i1>, tensor<1x2xf64>
          %1987 = stablehlo.select %1369, %1954, %iterArg_96 : tensor<i1>, tensor<1x2xf64>
          %1988 = stablehlo.select %1369, %1952, %iterArg_97 : tensor<i1>, tensor<1x2xf64>
          %1989 = stablehlo.maximum %iterArg_103, %1963 : tensor<f64>
          %1990 = stablehlo.subtract %iterArg_103, %1963 : tensor<f64>
          %1991 = stablehlo.compare  NE, %1990, %1990 : (tensor<f64>, tensor<f64>) -> tensor<i1>
          %1992 = stablehlo.add %iterArg_103, %1963 : tensor<f64>
          %1993 = stablehlo.abs %1990 : tensor<f64>
          %1994 = stablehlo.negate %1993 : tensor<f64>
          %1995 = stablehlo.exponential %1994 : tensor<f64>
          %1996 = stablehlo.log_plus_one %1995 : tensor<f64>
          %1997 = stablehlo.add %1989, %1996 : tensor<f64>
          %1998 = stablehlo.select %1991, %1992, %1997 : tensor<i1>, tensor<f64>
          %1999 = stablehlo.subtract %1963, %iterArg_103 : tensor<f64>
          %2000 = stablehlo.logistic %1999 : tensor<f64>
          %output_state_113, %output_114 = stablehlo.rng_bit_generator %1939, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
          %2001 = stablehlo.shift_right_logical %output_114, %c_26 : tensor<ui64>
          %2002 = stablehlo.or %2001, %c_25 : tensor<ui64>
          %2003 = stablehlo.bitcast_convert %2002 : (tensor<ui64>) -> tensor<f64>
          %2004 = stablehlo.subtract %2003, %cst_68 : tensor<f64>
          %2005 = stablehlo.compare  LT, %2004, %2000,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
          %2006 = stablehlo.select %2005, %1948, %iterArg_98 : tensor<i1>, tensor<1x2xf64>
          %2007 = stablehlo.select %2005, %1952, %iterArg_99 : tensor<i1>, tensor<1x2xf64>
          %2008 = stablehlo.select %2005, %1950, %iterArg_100 : tensor<i1>, tensor<f64>
          %2009 = stablehlo.select %2005, %1958, %iterArg_101 : tensor<i1>, tensor<f64>
          %2010 = stablehlo.add %iterArg_102, %c_66 : tensor<i64>
          %2011 = stablehlo.or %iterArg_105, %1964 : tensor<i1>
          %2012 = stablehlo.add %iterArg_106, %1966 : tensor<f64>
          %2013 = stablehlo.add %iterArg_107, %c_66 : tensor<i64>
          %2014 = stablehlo.add %iterArg_108, %1954 : tensor<1x2xf64>
          stablehlo.return %1983, %1984, %1985, %1986, %1987, %1988, %2006, %2007, %2008, %2009, %2010, %1998, %2011, %2012, %2013, %2014 : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>
        }) : (tensor<i1>) -> (tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>)
        %1969 = stablehlo.shift_right_logical %iterArg_112, %c_66 : tensor<i64>
        %1970 = stablehlo.popcnt %1969 : tensor<i64>
        %1971 = stablehlo.add %iterArg_112, %c_66 : tensor<i64>
        %1972 = stablehlo.not %iterArg_112 : tensor<i64>
        %1973 = stablehlo.and %1972, %1971 : tensor<i64>
        %1974 = stablehlo.subtract %1973, %c_66 : tensor<i64>
        %1975 = stablehlo.popcnt %1974 : tensor<i64>
        %1976 = stablehlo.subtract %1970, %1975 : tensor<i64>
        %1977 = stablehlo.add %1976, %c_66 : tensor<i64>
        %1978 = stablehlo.and %iterArg_112, %c_66 : tensor<i64>
        %1979 = stablehlo.compare  EQ, %1978, %c_67,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %1980:2 = "stablehlo.if"(%1979) ({
          %1983 = stablehlo.dynamic_update_slice %iterArg_110, %1954, %1970, %c_67 : (tensor<3x2xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<3x2xf64>
          %1984 = stablehlo.dynamic_update_slice %iterArg_111, %1968#15, %1970, %c_67 : (tensor<3x2xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<3x2xf64>
          stablehlo.return %1983, %1984 : tensor<3x2xf64>, tensor<3x2xf64>
        }, {
          stablehlo.return %iterArg_110, %iterArg_111 : tensor<3x2xf64>, tensor<3x2xf64>
        }) : (tensor<i1>) -> (tensor<3x2xf64>, tensor<3x2xf64>)
        %1981:2 = stablehlo.while(%iterArg_113 = %1970, %iterArg_114 = %c_63) : tensor<i64>, tensor<i1>
        cond {
          %1983 = stablehlo.compare  GE, %iterArg_113, %1977,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
          %1984 = stablehlo.not %iterArg_114 : tensor<i1>
          %1985 = stablehlo.and %1983, %1984 : tensor<i1>
          stablehlo.return %1985 : tensor<i1>
        } do {
          %1983 = stablehlo.dynamic_slice %1980#0, %iterArg_113, %c_67, sizes = [1, 2] : (tensor<3x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
          %1984 = stablehlo.dynamic_slice %1980#1, %iterArg_113, %c_67, sizes = [1, 2] : (tensor<3x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
          %1985 = stablehlo.subtract %1968#15, %1984 : tensor<1x2xf64>
          %1986 = stablehlo.add %1985, %1983 : tensor<1x2xf64>
          %1987 = stablehlo.dot_general %1983, %arg3, contracting_dims = [1] x [1] : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
          %1988 = stablehlo.add %1983, %1954 : tensor<1x2xf64>
          %1989 = stablehlo.multiply %cst, %1988 : tensor<1x2xf64>
          %1990 = stablehlo.add %1986, %1989 : tensor<1x2xf64>
          %1991 = stablehlo.dot_general %1987, %1990, contracting_dims = [0, 1] x [0, 1] : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
          %1992 = stablehlo.dot_general %1955, %1990, contracting_dims = [0, 1] x [0, 1] : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
          %1993 = stablehlo.compare  LE, %1991, %cst_65,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
          %1994 = stablehlo.compare  LE, %1992, %cst_65,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
          %1995 = stablehlo.or %1993, %1994 : tensor<i1>
          %1996 = stablehlo.subtract %iterArg_113, %c_66 : tensor<i64>
          stablehlo.return %1996, %1995 : tensor<i64>, tensor<i1>
        }
        %1982 = stablehlo.select %1967, %c_63, %1981#1 : tensor<i1>, tensor<i1>
        stablehlo.return %1968#0, %1968#1, %1968#2, %1968#3, %1968#4, %1968#5, %1968#6, %1968#7, %1968#8, %1968#9, %1968#10, %1968#11, %1982, %1968#12, %1968#13, %1968#14, %1968#15, %1932, %1980#0, %1980#1, %1971 : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<i64>, tensor<1x2xf64>, tensor<2xui64>, tensor<3x2xf64>, tensor<3x2xf64>, tensor<i64>
      }
      %1632 = stablehlo.select %1369, %iterArg, %1631#0 : tensor<i1>, tensor<1x2xf64>
      %1633 = stablehlo.select %1369, %iterArg_72, %1631#1 : tensor<i1>, tensor<1x2xf64>
      %1634 = stablehlo.select %1369, %iterArg_73, %1631#2 : tensor<i1>, tensor<1x2xf64>
      %1635 = stablehlo.select %1369, %1631#3, %iterArg_74 : tensor<i1>, tensor<1x2xf64>
      %1636 = stablehlo.select %1369, %1631#4, %iterArg_75 : tensor<i1>, tensor<1x2xf64>
      %1637 = stablehlo.select %1369, %1631#5, %iterArg_76 : tensor<i1>, tensor<1x2xf64>
      %1638 = stablehlo.maximum %iterArg_82, %1631#11 : tensor<f64>
      %1639 = stablehlo.subtract %iterArg_82, %1631#11 : tensor<f64>
      %1640 = stablehlo.compare  NE, %1639, %1639 : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %1641 = stablehlo.add %iterArg_82, %1631#11 : tensor<f64>
      %1642 = stablehlo.abs %1639 : tensor<f64>
      %1643 = stablehlo.negate %1642 : tensor<f64>
      %1644 = stablehlo.exponential %1643 : tensor<f64>
      %1645 = stablehlo.log_plus_one %1644 : tensor<f64>
      %1646 = stablehlo.add %1638, %1645 : tensor<f64>
      %1647 = stablehlo.select %1640, %1641, %1646 : tensor<i1>, tensor<f64>
      %1648 = stablehlo.subtract %1631#11, %iterArg_82 : tensor<f64>
      %1649 = stablehlo.exponential %1648 : tensor<f64>
      %1650 = stablehlo.minimum %1649, %cst_68 : tensor<f64>
      %1651 = stablehlo.or %1631#12, %1631#13 : tensor<i1>
      %1652 = stablehlo.select %1651, %cst_65, %1650 : tensor<i1>, tensor<f64>
      %output_state_90, %output_91 = stablehlo.rng_bit_generator %1629, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
      %1653 = stablehlo.shift_right_logical %output_91, %c_26 : tensor<ui64>
      %1654 = stablehlo.or %1653, %c_25 : tensor<ui64>
      %1655 = stablehlo.bitcast_convert %1654 : (tensor<ui64>) -> tensor<f64>
      %1656 = stablehlo.subtract %1655, %cst_68 : tensor<f64>
      %1657 = stablehlo.compare  LT, %1656, %1652,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %1658 = stablehlo.select %1657, %1631#6, %iterArg_77 : tensor<i1>, tensor<1x2xf64>
      %1659 = stablehlo.select %1657, %1631#7, %iterArg_78 : tensor<i1>, tensor<1x2xf64>
      %1660 = stablehlo.select %1657, %1631#8, %iterArg_79 : tensor<i1>, tensor<f64>
      %1661 = stablehlo.select %1657, %1631#9, %iterArg_80 : tensor<i1>, tensor<f64>
      %1662 = stablehlo.add %iterArg_81, %c_66 : tensor<i64>
      %1663 = stablehlo.add %iterArg_86, %1631#16 : tensor<1x2xf64>
      %1664 = stablehlo.dot_general %1633, %arg3, contracting_dims = [1] x [1] : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
      %1665 = stablehlo.dot_general %1636, %arg3, contracting_dims = [1] x [1] : (tensor<1x2xf64>, tensor<2x2xf64>) -> tensor<1x2xf64>
      %1666 = stablehlo.add %1633, %1636 : tensor<1x2xf64>
      %1667 = stablehlo.multiply %cst, %1666 : tensor<1x2xf64>
      %1668 = stablehlo.add %1663, %1667 : tensor<1x2xf64>
      %1669 = stablehlo.dot_general %1664, %1668, contracting_dims = [0, 1] x [0, 1] : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
      %1670 = stablehlo.dot_general %1665, %1668, contracting_dims = [0, 1] x [0, 1] : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<f64>
      %1671 = stablehlo.compare  LE, %1669, %cst_65,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %1672 = stablehlo.compare  LE, %1670, %cst_65,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %1673 = stablehlo.or %1671, %1672 : tensor<i1>
      %1674 = stablehlo.or %1631#12, %1673 : tensor<i1>
      %1675 = stablehlo.or %iterArg_84, %1631#13 : tensor<i1>
      %1676 = stablehlo.add %iterArg_85, %1631#14 : tensor<f64>
      stablehlo.return %1632, %1633, %1634, %1635, %1636, %1637, %1658, %1659, %1660, %1661, %1662, %1647, %1674, %1675, %1676, %1663, %1350 : tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<1x2xf64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>, tensor<i1>, tensor<i1>, tensor<f64>, tensor<1x2xf64>, tensor<2xui64>
    }
    %1097 = stablehlo.reshape %1096#6 : (tensor<1x2xf64>) -> tensor<2x1xf64>
    return %1097, %1067 : tensor<2x1xf64>, tensor<2xui64>
  }
}